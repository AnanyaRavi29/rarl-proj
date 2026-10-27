from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct

from gta_jax_w.data import Batch, ReplayBuffer
from gta_jax_w.rl.networks import Actor, Critic


DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
ALPHA = 2.5
LEARNING_RATE = 3e-4
MAX_ACTION = 1.0


@struct.dataclass
class TD3BCState:
    actor_params: object
    actor_target_params: object

    critic1_params: object
    critic1_target_params: object

    critic2_params: object
    critic2_target_params: object

    actor_opt_state: object
    critic1_opt_state: object
    critic2_opt_state: object

    rng: jax.Array
    step: jax.Array
    last_actor_loss: jax.Array


def soft_update(target, source):
    return jax.tree_util.tree_map(
        lambda t, s: (1.0 - TAU) * t + TAU * s,
        target,
        source,
    )


def build_td3bc(
    state_dim: int,
    action_dim: int,
    seed: int = 0,
    batch_size: int = 1024,
):
    actor = Actor(
        action_dim=action_dim,
        max_action=MAX_ACTION,
    )

    critic1 = Critic()
    critic2 = Critic()

    actor_tx = optax.adam(LEARNING_RATE)
    critic1_tx = optax.adam(LEARNING_RATE)
    critic2_tx = optax.adam(LEARNING_RATE)

    key = jax.random.PRNGKey(seed)
    key, actor_key, q1_key, q2_key = jax.random.split(key, 4)

    dummy_obs = jnp.zeros((1, state_dim), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

    actor_params = actor.init(
        actor_key,
        dummy_obs,
    )["params"]

    critic1_params = critic1.init(
        q1_key,
        dummy_obs,
        dummy_action,
    )["params"]

    critic2_params = critic2.init(
        q2_key,
        dummy_obs,
        dummy_action,
    )["params"]

    state = TD3BCState(
        actor_params=actor_params,
        actor_target_params=actor_params,

        critic1_params=critic1_params,
        critic1_target_params=critic1_params,

        critic2_params=critic2_params,
        critic2_target_params=critic2_params,

        actor_opt_state=actor_tx.init(actor_params),
        critic1_opt_state=critic1_tx.init(critic1_params),
        critic2_opt_state=critic2_tx.init(critic2_params),

        rng=key,
        step=jnp.array(0, dtype=jnp.int32),
        last_actor_loss=jnp.array(0.0, dtype=jnp.float32),
    )

    def train_step(state, buffer):
        rng, sample_key, noise_key = jax.random.split(
            state.rng,
            3,
        )

        indices = jax.random.randint(
            sample_key,
            shape=(batch_size,),
            minval=0,
            maxval=buffer.observations.shape[0],
        )

        batch = Batch(
            observations=buffer.observations[indices],
            actions=buffer.actions[indices],
            rewards=buffer.rewards[indices],
            next_observations=buffer.next_observations[indices],
            terminals=buffer.terminals[indices],
            weights=buffer.weights[indices],
        )
        w = batch.weights
        wsum = jnp.sum(w) + 1e-8

        # -----------------------------------
        # TD3 target action
        # -----------------------------------

        noise = jax.random.normal(
            noise_key,
            shape=batch.actions.shape,
        )

        noise = jnp.clip(
            noise * POLICY_NOISE,
            -NOISE_CLIP,
            NOISE_CLIP,
        )

        next_action = actor.apply(
            {"params": state.actor_target_params},
            batch.next_observations,
        )

        next_action = jnp.clip(
            next_action + noise,
            -MAX_ACTION,
            MAX_ACTION,
        )

        target_q1 = critic1.apply(
            {"params": state.critic1_target_params},
            batch.next_observations,
            next_action,
        )

        target_q2 = critic2.apply(
            {"params": state.critic2_target_params},
            batch.next_observations,
            next_action,
        )

        target_q = jnp.minimum(
            target_q1,
            target_q2,
        )

        target_q = (
            batch.rewards
            + (1.0 - batch.terminals)
            * DISCOUNT
            * target_q
        )

        target_q = jax.lax.stop_gradient(target_q)

        # -----------------------------------
        # Twin critic update
        # -----------------------------------

        def critic_loss_fn(q1_params, q2_params):
            q1 = critic1.apply(
                {"params": q1_params},
                batch.observations,
                batch.actions,
            )

            q2 = critic2.apply(
                {"params": q2_params},
                batch.observations,
                batch.actions,
            )

            q1_loss = jnp.sum(w * jnp.square(q1 - target_q)) / wsum
            q2_loss = jnp.sum(w * jnp.square(q2 - target_q)) / wsum

            return q1_loss + q2_loss, (
                jnp.mean(q1),
                jnp.mean(q2),
            )

        (
            (critic_loss, (q1_mean, q2_mean)),
            (q1_grads, q2_grads),
        ) = jax.value_and_grad(
            critic_loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(
            state.critic1_params,
            state.critic2_params,
        )

        q1_updates, q1_opt_state = critic1_tx.update(
            q1_grads,
            state.critic1_opt_state,
            state.critic1_params,
        )

        q2_updates, q2_opt_state = critic2_tx.update(
            q2_grads,
            state.critic2_opt_state,
            state.critic2_params,
        )

        q1_params = optax.apply_updates(
            state.critic1_params,
            q1_updates,
        )

        q2_params = optax.apply_updates(
            state.critic2_params,
            q2_updates,
        )

        new_step = state.step + 1

        # -----------------------------------
        # Delayed actor + target updates
        # -----------------------------------

        def update_actor(_):
            def actor_loss_fn(actor_params):
                pi = actor.apply(
                    {"params": actor_params},
                    batch.observations,
                )

                q = critic1.apply(
                    {"params": q1_params},
                    batch.observations,
                    pi,
                )

                # TD3+BC scaling coefficient
                lmbda = ALPHA / jnp.mean(jnp.abs(q))
                lmbda = jax.lax.stop_gradient(lmbda)

                bc_loss = jnp.sum(w * jnp.square(pi - batch.actions)) / (wsum * pi.shape[-1])

                actor_loss = (
                    -lmbda * jnp.sum(w * q) / wsum
                    + bc_loss
                )

                return actor_loss

            actor_loss, actor_grads = jax.value_and_grad(
                actor_loss_fn
            )(state.actor_params)

            actor_updates, actor_opt_state = actor_tx.update(
                actor_grads,
                state.actor_opt_state,
                state.actor_params,
            )

            actor_params = optax.apply_updates(
                state.actor_params,
                actor_updates,
            )

            actor_target_params = soft_update(
                state.actor_target_params,
                actor_params,
            )

            q1_target_params = soft_update(
                state.critic1_target_params,
                q1_params,
            )

            q2_target_params = soft_update(
                state.critic2_target_params,
                q2_params,
            )

            return (
                actor_params,
                actor_target_params,
                actor_opt_state,
                q1_target_params,
                q2_target_params,
                actor_loss,
            )

        def skip_actor(_):
            return (
                state.actor_params,
                state.actor_target_params,
                state.actor_opt_state,
                state.critic1_target_params,
                state.critic2_target_params,
                state.last_actor_loss,
            )

        (
            actor_params,
            actor_target_params,
            actor_opt_state,
            q1_target_params,
            q2_target_params,
            actor_loss,
        ) = jax.lax.cond(
            new_step % POLICY_FREQ == 0,
            update_actor,
            skip_actor,
            operand=None,
        )

        new_state = TD3BCState(
            actor_params=actor_params,
            actor_target_params=actor_target_params,

            critic1_params=q1_params,
            critic1_target_params=q1_target_params,

            critic2_params=q2_params,
            critic2_target_params=q2_target_params,

            actor_opt_state=actor_opt_state,
            critic1_opt_state=q1_opt_state,
            critic2_opt_state=q2_opt_state,

            rng=rng,
            step=new_step,
            last_actor_loss=actor_loss,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q1": q1_mean,
            "q2": q2_mean,
        }

        return new_state, metrics

    @partial(
        jax.jit,
        static_argnames=("num_steps",),
    )
    def run_chunk(state, buffer, num_steps):
        def body(carry, _):
            carry, metrics = train_step(
                carry,
                buffer,
            )

            return carry, metrics

        state, metrics = jax.lax.scan(
            body,
            state,
            xs=None,
            length=num_steps,
        )

        last_metrics = jax.tree_util.tree_map(
            lambda x: x[-1],
            metrics,
        )

        return state, last_metrics

    return state, run_chunk

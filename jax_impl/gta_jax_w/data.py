from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class Batch:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    weights: jax.Array


@struct.dataclass
class ReplayBuffer:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    weights: jax.Array

    @property
    def size(self):
        return self.observations.shape[0]


def load_qlearning_dataset(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Loading dataset: {path}")

    with np.load(path) as data:
        observations = data["observations"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        next_observations = data["next_observations"].astype(np.float32)
        terminals = data["terminals"].astype(np.float32)
        weights = (data["weights"] if "weights" in data.files else np.ones(len(terminals))).astype(np.float32)

    # TD3+BC standard state normalization.
    state_mean = observations.mean(axis=0, keepdims=True)
    state_std = observations.std(axis=0, keepdims=True) + 1e-3

    observations = (observations - state_mean) / state_std
    next_observations = (next_observations - state_mean) / state_std

    # Use [N, 1] for scalar transition quantities.
    rewards = rewards.reshape(-1, 1)
    terminals = terminals.reshape(-1, 1)

    print("Moving replay buffer to JAX device...")

    replay_buffer = ReplayBuffer(
        observations=jax.device_put(jnp.asarray(observations)),
        actions=jax.device_put(jnp.asarray(actions)),
        rewards=jax.device_put(jnp.asarray(rewards)),
        next_observations=jax.device_put(jnp.asarray(next_observations)),
        terminals=jax.device_put(jnp.asarray(terminals)),
        weights=jax.device_put(jnp.asarray(weights.reshape(-1, 1))),
    )

    # Synchronize so loading timing/output is honest.
    jax.block_until_ready(replay_buffer.observations)

    stats = {
        "state_mean": state_mean.squeeze(0),
        "state_std": state_std.squeeze(0),
        "num_transitions": observations.shape[0],
        "state_dim": observations.shape[1],
        "action_dim": actions.shape[1],
    }

    return replay_buffer, stats


@jax.jit(static_argnames=("batch_size",))
def sample_batch(
    buffer: ReplayBuffer,
    key: jax.Array,
    batch_size: int = 1024,
):
    indices = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=buffer.observations.shape[0],
    )

    return Batch(
        observations=buffer.observations[indices],
        actions=buffer.actions[indices],
        rewards=buffer.rewards[indices],
        next_observations=buffer.next_observations[indices],
        terminals=buffer.terminals[indices],
        weights=buffer.weights[indices],
    )

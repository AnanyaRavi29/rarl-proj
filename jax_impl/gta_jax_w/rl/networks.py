from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def torch_uniform_init(fan_in: int):
    """Approximate PyTorch nn.Linear default initialization."""
    bound = 1.0 / jnp.sqrt(float(fan_in))

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key,
            shape,
            dtype,
            minval=-bound,
            maxval=bound,
        )

    return init


class Actor(nn.Module):
    action_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations):
        x = observations

        for hidden_dim in self.hidden_dims:
            fan_in = x.shape[-1]
            x = nn.Dense(
                hidden_dim,
                kernel_init=torch_uniform_init(fan_in),
                bias_init=torch_uniform_init(fan_in),
            )(x)
            x = nn.relu(x)

        fan_in = x.shape[-1]
        x = nn.Dense(
            self.action_dim,
            kernel_init=torch_uniform_init(fan_in),
            bias_init=torch_uniform_init(fan_in),
        )(x)

        return self.max_action * jnp.tanh(x)


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        for hidden_dim in self.hidden_dims:
            fan_in = x.shape[-1]
            x = nn.Dense(
                hidden_dim,
                kernel_init=torch_uniform_init(fan_in),
                bias_init=torch_uniform_init(fan_in),
            )(x)
            x = nn.relu(x)

        fan_in = x.shape[-1]
        return nn.Dense(
            1,
            kernel_init=torch_uniform_init(fan_in),
            bias_init=torch_uniform_init(fan_in),
        )(x)


def count_parameters(params):
    return sum(
        int(x.size)
        for x in jax.tree_util.tree_leaves(params)
    )

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def torch_uniform_init(bound: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key,
            shape,
            dtype=dtype,
            minval=-bound,
            maxval=bound,
        )
    return init


class TorchDense(nn.Module):
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        fan_in = int(x.shape[-1])
        bound = 1.0 / math.sqrt(fan_in)

        return nn.Dense(
            self.features,
            use_bias=self.use_bias,
            kernel_init=torch_uniform_init(bound),
            bias_init=torch_uniform_init(bound),
        )(x)


class TorchConv2D(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] = (1, 1)
    padding: str = "SAME"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        in_channels = int(x.shape[-1])
        kernel_area = int(
            self.kernel_size[0]
            * self.kernel_size[1]
        )

        fan_in = in_channels * kernel_area
        bound = 1.0 / math.sqrt(fan_in)

        return nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=torch_uniform_init(bound),
            bias_init=torch_uniform_init(bound),
        )(x)


class TorchConvTranspose2D(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int]
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        # PyTorch ConvTranspose2d fan_in uses output channels.
        kernel_area = int(
            self.kernel_size[0]
            * self.kernel_size[1]
        )

        fan_in = self.features * kernel_area
        bound = 1.0 / math.sqrt(fan_in)

        return nn.ConvTranspose(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_init=torch_uniform_init(bound),
            bias_init=torch_uniform_init(bound),
        )(x)


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        if x.ndim == 1:
            x = x[:, None]

        half_dim = self.dim // 2

        scale = math.log(10000.0) / (
            half_dim - 1
        )

        emb = jnp.exp(
            jnp.arange(
                half_dim,
                dtype=x.dtype,
            ) * -scale
        )

        emb = x * emb[None, :]

        return jnp.concatenate(
            [
                jnp.sin(emb),
                jnp.cos(emb),
            ],
            axis=-1,
        )


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    dim: int
    is_random: bool = False

    @nn.compact
    def __call__(self, x):
        if x.ndim == 1:
            x = x[:, None]

        assert self.dim % 2 == 0

        half_dim = self.dim // 2

        def random_normal(key, shape):
            return jax.random.normal(
                key,
                shape,
                dtype=jnp.float32,
            )

        if self.is_random:
            weights = self.variable(
        "constants",
        "weights",
        lambda: jax.random.normal(
            self.make_rng("params"),
            (half_dim,),
            dtype=jnp.float32,
        ),
    ).value
        else:
            weights = self.param(
                "weights",
                random_normal,
                (half_dim,),
            )

        freqs = (
            x
            * weights[None, :]
            * 2.0
            * math.pi
        )

        fouriered = jnp.concatenate(
            [
                jnp.sin(freqs),
                jnp.cos(freqs),
            ],
            axis=-1,
        )

        return jnp.concatenate(
            [x, fouriered],
            axis=-1,
        )


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = TorchDense(
            self.hidden_dim,
            name="fc1",
        )(x)

        # Match PyTorch nn.GELU default.
        x = jax.nn.gelu(
            x,
            approximate=False,
        )

        x = TorchDense(
            self.dim,
            name="fc2",
        )(x)

        return x


class InstanceNorm2D(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]

        scale = self.param(
            "scale",
            nn.initializers.ones,
            (channels,),
        )

        bias = self.param(
            "bias",
            nn.initializers.zeros,
            (channels,),
        )

        mean = jnp.mean(
            x,
            axis=(1, 2),
            keepdims=True,
        )

        var = jnp.var(
            x,
            axis=(1, 2),
            keepdims=True,
        )

        x = (
            (x - mean)
            / jnp.sqrt(var + self.epsilon)
        )

        return (
            x * scale[None, None, None, :]
            + bias[None, None, None, :]
        )


class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        b, h, w, _ = x.shape

        hidden_dim = (
            self.heads * self.dim_head
        )

        qkv = TorchConv2D(
            features=hidden_dim * 3,
            kernel_size=(1, 1),
            use_bias=False,
            name="to_qkv",
        )(x)

        qkv = qkv.reshape(
            b,
            h,
            w,
            3,
            self.heads,
            self.dim_head,
        )

        qkv = jnp.transpose(
            qkv,
            (3, 0, 4, 5, 1, 2),
        )

        qkv = qkv.reshape(
            3,
            b,
            self.heads,
            self.dim_head,
            h * w,
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        k = jax.nn.softmax(
            k,
            axis=-1,
        )

        context = jnp.einsum(
            "bhdn,bhen->bhde",
            k,
            v,
        )

        out = jnp.einsum(
            "bhde,bhdn->bhen",
            context,
            q,
        )

        out = out.reshape(
            b,
            self.heads,
            self.dim_head,
            h,
            w,
        )

        out = jnp.transpose(
            out,
            (0, 3, 4, 1, 2),
        )

        out = out.reshape(
            b,
            h,
            w,
            hidden_dim,
        )

        return TorchConv2D(
            features=self.dim,
            kernel_size=(1, 1),
            name="to_out",
        )(out)


class PreNormAttention(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        residual = x

        x = InstanceNorm2D(
            epsilon=1e-5,
            name="norm",
        )(x)

        x = LinearAttention(
            self.dim,
            name="attention",
        )(x)

        return x + residual


class Block(nn.Module):
    dim_out: int
    groups: int = 8

    @nn.compact
    def __call__(self, x):
        x = TorchConv2D(
            self.dim_out,
            kernel_size=(3, 3),
            name="conv",
        )(x)

        x = nn.GroupNorm(
            num_groups=self.groups,
            epsilon=1e-5,
            name="group_norm",
        )(x)

        return mish(x)


class Downsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        return TorchConv2D(
            features=self.dim,
            kernel_size=(3, 3),
            strides=(2, 1),
            padding="SAME",
            name="conv",
        )(x)


class Upsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        return TorchConvTranspose2D(
            features=self.dim,
            kernel_size=(4, 3),
            strides=(2, 1),
            padding="SAME",
            name="conv",
        )(x)

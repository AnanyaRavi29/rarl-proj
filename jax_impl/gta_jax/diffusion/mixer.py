from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from gta_jax.diffusion.helpers import (
    Block,
    Downsample,
    FeedForward,
    PreNormAttention,
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
    TorchConv2D,
    TorchDense,
    Upsample,
    mish,
)


class MixerBlock(nn.Module):
    height: int
    width: int
    dim: int
    dim_out: int
    time_emb_dim: int

    @nn.compact
    def __call__(
        self,
        x,
        time_emb,
    ):
        # JAX representation:
        # [batch, height, width, channels]
        b, height, width, channels = x.shape

        assert height == self.height
        assert width == self.width
        assert channels == self.dim

        # ----------------------------------
        # Mixer over channel * joined width
        # ----------------------------------

        # PyTorch:
        # NCHW -> N H C W -> N H (C*W)
        x_first = jnp.transpose(
            x,
            (0, 1, 3, 2),
        )

        x_first = x_first.reshape(
            b,
            height,
            channels * width,
        )

        x_first = nn.LayerNorm(
            epsilon=1e-5,
            name="ln1",
        )(x_first)

        x_first = FeedForward(
            dim=channels * width,
            hidden_dim=4 * self.dim,
            name="block1",
        )(x_first)

        x_first = x_first.reshape(
            b,
            height,
            channels,
            width,
        )

        x_first = jnp.transpose(
            x_first,
            (0, 1, 3, 2),
        )

        h = x_first

        # ----------------------------------
        # Mixer over trajectory horizon
        # ----------------------------------

        # Equivalent to PyTorch:
        # N C H W -> N C W H
        x_second = jnp.transpose(
            x,
            (0, 3, 2, 1),
        )

        x_second = nn.LayerNorm(
            epsilon=1e-5,
            name="ln2",
        )(x_second)

        x_second = FeedForward(
            dim=height,
            hidden_dim=4 * height,
            name="block2",
        )(x_second)

        x_second = jnp.transpose(
            x_second,
            (0, 3, 2, 1),
        )

        h = h + x_second

        if self.dim != self.dim_out:
            h = TorchConv2D(
                self.dim_out,
                kernel_size=(1, 1),
                name="output_conv",
            )(h)

            residual = TorchConv2D(
                self.dim_out,
                kernel_size=(1, 1),
                name="res_conv",
            )(x)
        else:
            residual = x

        time_projection = TorchDense(
            self.dim_out,
            name="time_projection",
        )(mish(time_emb))

        h = (
            h
            + time_projection[
                :,
                None,
                None,
                :,
            ]
        )

        return h + residual


class MixerUnet(nn.Module):
    horizon: int
    d_in: int
    cond_dim: int

    dim: int = 16
    dim_mults: Sequence[int] = (
        1,
        2,
        4,
        8,
    )

    channels: int = 1

    condition_dropout: float = 0.25
    use_dropout: bool = True
    force_dropout: bool = False

    learned_sinusoidal_dim: int = 16
    learned_sinusoidal_cond: bool = False
    random_fourier_features: bool = True

    @nn.compact
    def __call__(
        self,
        x,
        time,
        cond=None,
        *,
        deterministic=False,
    ):
        if time.ndim == 1:
            time = time[:, None]

        # ----------------------------------
        # Time embedding
        # ----------------------------------

        random_or_learned = (
            self.learned_sinusoidal_cond
            or self.random_fourier_features
        )

        if random_or_learned:
            t = RandomOrLearnedSinusoidalPosEmb(
                self.learned_sinusoidal_dim,
                is_random=self.random_fourier_features,
                name="time_fourier",
            )(time)

            fourier_dim = (
                self.learned_sinusoidal_dim
                + 1
            )
        else:
            t = SinusoidalPosEmb(
                self.dim,
                name="time_sinusoidal",
            )(time)

            fourier_dim = self.dim

        assert t.shape[-1] == fourier_dim

        t = TorchDense(
            self.dim * 4,
            name="time_fc1",
        )(t)

        t = mish(t)

        t = TorchDense(
            self.dim,
            name="time_fc2",
        )(t)

        time_dim = self.dim

        # ----------------------------------
        # Return conditioning
        # ----------------------------------

        if self.cond_dim > 0:
            if cond is not None:
                if cond.ndim == 1:
                    cond = cond[:, None]

                cond_emb = TorchDense(
                    self.dim * 2,
                    name="cond_fc1",
                )(cond)

                cond_emb = mish(cond_emb)

                cond_emb = TorchDense(
                    self.dim,
                    name="cond_fc2",
                )(cond_emb)

                if (
                    self.use_dropout
                    and not deterministic
                ):
                    mask = jax.random.bernoulli(
                        self.make_rng("dropout"),
                        p=1.0
                        - self.condition_dropout,
                        shape=(
                            cond_emb.shape[0],
                            1,
                        ),
                    )

                    cond_emb = (
                        cond_emb
                        * mask.astype(
                            cond_emb.dtype
                        )
                    )

                if self.force_dropout:
                    cond_emb = (
                        jnp.zeros_like(
                            cond_emb
                        )
                    )
            else:
                cond_emb = jnp.zeros_like(t)

            t = jnp.concatenate(
                [cond_emb, t],
                axis=-1,
            )

            time_dim = 2 * self.dim

        # ----------------------------------
        # Add channel dimension
        # PyTorch: [B,T,D] -> [B,1,T,D]
        # JAX NHWC: [B,T,D] -> [B,T,D,1]
        # ----------------------------------

        x = x[..., None]

        dims = [
            self.channels,
            *[
                self.dim * m
                for m in self.dim_mults
            ],
        ]

        in_out = list(
            zip(
                dims[:-1],
                dims[1:],
            )
        )

        num_resolutions = len(in_out)

        is_seq = (
            self.horizon % (2 ** 3)
            == 0
        )

        current_horizon = (
            self.horizon
        )

        skips = []

        # ----------------------------------
        # Down path
        # ----------------------------------

        for ind, (
            dim_in,
            dim_out,
        ) in enumerate(in_out):
            is_last = (
                ind
                >= num_resolutions - 1
            )

            x = MixerBlock(
                height=current_horizon,
                width=self.d_in,
                dim=dim_in,
                dim_out=dim_out,
                time_emb_dim=time_dim,
                name=f"down_{ind}_block1",
            )(x, t)

            x = MixerBlock(
                height=current_horizon,
                width=self.d_in,
                dim=dim_out,
                dim_out=dim_out,
                time_emb_dim=time_dim,
                name=f"down_{ind}_block2",
            )(x, t)

            x = PreNormAttention(
                dim_out,
                name=f"down_{ind}_attn",
            )(x)

            skips.append(x)

            if is_seq and not is_last:
                x = Downsample(
                    dim_out,
                    name=f"down_{ind}_sample",
                )(x)

                current_horizon //= 2

        # ----------------------------------
        # Middle
        # ----------------------------------

        mid_dim = dims[-1]

        x = MixerBlock(
            height=current_horizon,
            width=self.d_in,
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=time_dim,
            name="mid_block1",
        )(x, t)

        x = PreNormAttention(
            mid_dim,
            name="mid_attn",
        )(x)

        x = MixerBlock(
            height=current_horizon,
            width=self.d_in,
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=time_dim,
            name="mid_block2",
        )(x, t)

        # ----------------------------------
        # Up path
        # ----------------------------------

        up_pairs = list(
            reversed(
                in_out[1:]
            )
        )

        for ind, (
            dim_in,
            dim_out,
        ) in enumerate(up_pairs):
            # This mirrors the original code
            # exactly. num_resolutions is 4,
            # while this loop has 3 stages.
            is_last = (
                ind
                >= num_resolutions - 1
            )

            x = jnp.concatenate(
                [
                    x,
                    skips.pop(),
                ],
                axis=-1,
            )

            x = MixerBlock(
                height=current_horizon,
                width=self.d_in,
                dim=dim_out * 2,
                dim_out=dim_in,
                time_emb_dim=time_dim,
                name=f"up_{ind}_block1",
            )(x, t)

            x = MixerBlock(
                height=current_horizon,
                width=self.d_in,
                dim=dim_in,
                dim_out=dim_in,
                time_emb_dim=time_dim,
                name=f"up_{ind}_block2",
            )(x, t)

            x = PreNormAttention(
                dim_in,
                name=f"up_{ind}_attn",
            )(x)

            if is_seq and not is_last:
                x = Upsample(
                    dim_in,
                    name=f"up_{ind}_sample",
                )(x)

                current_horizon *= 2

        # ----------------------------------
        # Final block
        # ----------------------------------

        x = Block(
            self.dim,
            groups=8,
            name="final_block",
        )(x)

        x = TorchConv2D(
            self.channels,
            kernel_size=(1, 1),
            name="final_conv",
        )(x)

        # NHWC -> [B,T,D]
        x = jnp.squeeze(
            x,
            axis=-1,
        )

        return x


def count_trainable_parameters(variables):
    return sum(
        int(x.size)
        for x
        in jax.tree_util.tree_leaves(
            variables["params"]
        )
    )


def count_constants(variables):
    if "constants" not in variables:
        return 0

    return sum(
        int(x.size)
        for x
        in jax.tree_util.tree_leaves(
            variables["constants"]
        )
    )

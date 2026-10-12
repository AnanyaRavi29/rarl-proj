import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

from flax import struct, serialization
from tqdm import tqdm

from gta_jax.diffusion.mixer import (
    MixerUnet,
    count_trainable_parameters,
)
from gta_jax.diffusion.edm import edm_loss


# ============================================================
# Utilities
# ============================================================

def format_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def discounted_cumsum(x, gamma):
    out = np.zeros_like(x, dtype=np.float32)
    out[-1] = x[-1]

    for t in range(len(x) - 2, -1, -1):
        out[t] = x[t] + gamma * out[t + 1]

    return out


# ============================================================
# Build cached GTA trajectory windows
# ============================================================

def prepare_dataset_cache(
    dataset_path,
    dataset_name,
    seq_len=31,
    gamma=0.99,
    reward_scale=0.01,
    penalty=100.0,
):
    stem = Path(dataset_path).stem
    data_dir = Path(dataset_path).parent

    windows_path = (
        data_dir / f"{stem}-diffusion-windows.npy"
    )
    cond_path = (
        data_dir / f"{stem}-diffusion-cond.npy"
    )
    norm_path = (
        data_dir / f"{stem}-diffusion-normalization.npz"
    )

    if (
        windows_path.exists()
        and cond_path.exists()
        and norm_path.exists()
    ):
        print("Using existing diffusion cache:")
        print(" ", windows_path)
        print(" ", cond_path)
        print(" ", norm_path)

        return windows_path, cond_path, norm_path

    print("=" * 70)
    print("PREPARING JAX DIFFUSION DATASET CACHE")
    print("=" * 70)

    with open(dataset_path, "rb") as f:
        episodes = pickle.load(f)

    obs_dim = 17
    action_dim = 6
    reward_dim = 1
    joined_dim = obs_dim + action_dim + reward_dim

    # --------------------------------------------------------
    # Exact GTA normalizer is calculated BEFORE terminal
    # penalty modification.
    # --------------------------------------------------------

    print("Computing GTA standard normalizer...")

    joined = []

    for episode in tqdm(
        episodes,
        desc="Normalizer",
        file=sys.stdout,
    ):
        obs = np.asarray(
            episode["observations"],
            dtype=np.float32,
        )

        actions = np.asarray(
            episode["actions"],
            dtype=np.float32,
        )

        rewards = np.asarray(
            episode["rewards"],
            dtype=np.float32,
        ).reshape(-1, 1)

        joined.append(
            np.concatenate(
                [obs, actions, rewards],
                axis=-1,
            )
        )

    joined = np.concatenate(
        joined,
        axis=0,
    )

    mean = joined.mean(
        axis=0
    ).astype(np.float32)

    # torch.std() uses sample standard deviation.
    std = (
        joined.std(
            axis=0,
            ddof=1,
        ).astype(np.float32)
        + 1e-5
    )

    del joined

    np.savez(
        norm_path,
        mean=mean,
        std=std,
    )

    # Count sliding sub-trajectories.
    num_windows = sum(
        max(
            0,
            len(ep["observations"])
            - seq_len
            + 1,
        )
        for ep in episodes
    )

    print()
    print("Trajectory windows :", num_windows)
    print("Expected near      : 970000")
    print("Window shape       :", (seq_len + 1, joined_dim))
    print()

    windows = np.lib.format.open_memmap(
        windows_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            num_windows,
            seq_len + 1,
            joined_dim,
        ),
    )

    conditions = np.lib.format.open_memmap(
        cond_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_windows, 1),
    )

    cursor = 0

    for episode in tqdm(
        episodes,
        desc="Trajectory cache",
        file=sys.stdout,
    ):
        obs = np.asarray(
            episode["observations"],
            dtype=np.float32,
        )

        actions = np.asarray(
            episode["actions"],
            dtype=np.float32,
        )

        rewards = np.asarray(
            episode["rewards"],
            dtype=np.float32,
        ).reshape(-1)

        next_obs = np.asarray(
            episode["next_observations"],
            dtype=np.float32,
        )

        terminals = np.asarray(
            episode["terminals"]
        ).astype(bool)

        length = len(obs)

        if length < seq_len:
            continue

        # Match DiffusionTrajectoryDataset.
        train_rewards = rewards.copy()

        if (
            "maze" not in dataset_name
            and terminals.any()
        ):
            if not terminals[-1]:
                raise RuntimeError(
                    "Terminal found before final transition."
                )

            train_rewards[-1] -= penalty

        returns = discounted_cumsum(
            train_rewards,
            gamma,
        )

        n = length - seq_len + 1

        starts = np.arange(
            n,
            dtype=np.int64,
        )

        indices = (
            starts[:, None]
            + np.arange(
                seq_len,
                dtype=np.int64,
            )[None, :]
        )

        block = np.zeros(
            (
                n,
                seq_len + 1,
                joined_dim,
            ),
            dtype=np.float32,
        )

        # First 31 transitions.
        block[
            :,
            :seq_len,
            :obs_dim
        ] = obs[indices]

        block[
            :,
            :seq_len,
            obs_dim:obs_dim + action_dim
        ] = actions[indices]

        block[
            :,
            :seq_len,
            obs_dim + action_dim
        ] = train_rewards[indices]

        # Final appended next state.
        last_indices = (
            starts
            + seq_len
            - 1
        )

        block[
            :,
            seq_len,
            :obs_dim
        ] = next_obs[last_indices]

        # action/reward remain zero on final row.

        windows[
            cursor:cursor + n
        ] = block

        conditions[
            cursor:cursor + n,
            0
        ] = (
            returns[starts]
            * reward_scale
        )

        cursor += n

    windows.flush()
    conditions.flush()

    if cursor != num_windows:
        raise RuntimeError(
            f"Expected {num_windows} windows, wrote {cursor}"
        )

    print()
    print("Diffusion cache complete.")
    print("Windows      :", windows_path)
    print("Conditions   :", cond_path)
    print("Normalization:", norm_path)

    return windows_path, cond_path, norm_path


# ============================================================
# Train state
# ============================================================

@struct.dataclass
class DiffusionTrainState:
    params: object
    ema_params: object
    opt_state: object
    rng: jax.Array
    step: jax.Array


def make_lr_schedule(
    base_lr,
    restart_period,
):
    # Equivalent to PyTorch:
    # CosineAnnealingWarmRestarts(T_0=100000)
    def schedule(step):
        position = jnp.mod(
            step,
            restart_period,
        )

        ratio = (
            position
            / float(restart_period)
        )

        return (
            base_lr
            * 0.5
            * (
                1.0
                + jnp.cos(
                    jnp.pi * ratio
                )
            )
        )

    return schedule


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/halfcheetah-medium-v2.pkl",
    )

    parser.add_argument(
        "--dataset-name",
        default="halfcheetah-medium-v2",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--run-name",
        default="diffusion_halfcheetah_medium_seed42",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("JAX GTA DIFFUSION")
    print("=" * 70)
    print("JAX version :", jax.__version__)
    print("Backend     :", jax.default_backend())
    print("Devices     :", jax.devices())
    print()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    windows_path, cond_path, norm_path = (
        prepare_dataset_cache(
            args.dataset,
            args.dataset_name,
        )
    )

    windows = np.load(
        windows_path,
        mmap_mode="r",
    )

    conditions = np.load(
        cond_path,
        mmap_mode="r",
    )

    norm = np.load(
        norm_path,
    )

    mean = jax.device_put(
        norm["mean"]
    )

    std = jax.device_put(
        norm["std"]
    )

    num_samples = windows.shape[0]

    print()
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print("Samples       :", num_samples)
    print("Input shape   :", windows.shape[1:])
    print("Batch size    :", args.batch_size)
    print("Steps         :", args.steps)
    print("Learning rate :", args.lr)
    print("Seed          :", args.seed)
    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MixerUnet(
        horizon=32,
        d_in=24,
        cond_dim=1,
        dim=16,
        condition_dropout=0.25,
        use_dropout=True,
        force_dropout=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=True,
        learned_sinusoidal_dim=16,
    )

    key = jax.random.PRNGKey(
        args.seed
    )

    key, init_key, train_key = (
        jax.random.split(
            key,
            3,
        )
    )

    dummy_x = jnp.zeros(
        (1, 32, 24),
        dtype=jnp.float32,
    )

    dummy_sigma = jnp.ones(
        (1, 1),
        dtype=jnp.float32,
    )

    dummy_cond = jnp.zeros(
        (1, 1),
        dtype=jnp.float32,
    )

    print("Initializing MixerUnet...")

    variables = model.init(
        {
            "params": init_key,
        },
        dummy_x,
        dummy_sigma,
        dummy_cond,
        deterministic=True,
    )

    params = variables["params"]
    constants = variables["constants"]

    param_count = (
        count_trainable_parameters(
            variables
        )
    )

    print(
        "Trainable parameters:",
        f"{param_count:,}",
    )

    if param_count != 29_628_949:
        raise RuntimeError(
            "MixerUnet parameter count mismatch."
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    restart_period = (
        args.steps // 10
    )

    lr_schedule = make_lr_schedule(
        args.lr,
        restart_period,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(
            1.0
        ),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.99,
            eps=1e-8,
            weight_decay=0.0,
        ),
    )

    state = DiffusionTrainState(
        params=params,
        ema_params=params,
        opt_state=optimizer.init(params),
        rng=train_key,
        step=jnp.array(
            0,
            dtype=jnp.int32,
        ),
    )

    ema_decay = 0.995
    ema_every = 10

    # --------------------------------------------------------
    # JIT training step
    # --------------------------------------------------------

    @jax.jit
    def train_step(
        state,
        batch,
        cond,
    ):
        next_rng, loss_rng = (
            jax.random.split(
                state.rng
            )
        )

        def loss_fn(params):
            loss, metrics = edm_loss(
                model=model,
                params=params,
                constants=constants,
                inputs=batch,
                cond=cond,
                mean=mean,
                std=std,
                rng=loss_rng,
                obs_dim=17,
            )

            return loss, metrics

        (
            loss,
            metrics,
        ), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)

        grad_norm = optax.global_norm(
            grads
        )

        updates, new_opt_state = (
            optimizer.update(
                grads,
                state.opt_state,
                state.params,
            )
        )

        new_params = optax.apply_updates(
            state.params,
            updates,
        )

        new_step = state.step + 1

        def update_ema(_):
            return jax.tree_util.tree_map(
                lambda ema, p:
                    ema_decay * ema
                    + (1.0 - ema_decay) * p,
                state.ema_params,
                new_params,
            )

        new_ema_params = jax.lax.cond(
            (
                new_step
                % ema_every
            ) == 0,
            update_ema,
            lambda _:
                state.ema_params,
            operand=None,
        )

        new_state = DiffusionTrainState(
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            rng=next_rng,
            step=new_step,
        )

        metrics = dict(metrics)

        metrics["grad_norm"] = (
            grad_norm
        )

        metrics["lr"] = (
            lr_schedule(
                state.step
            )
        )

        return new_state, metrics

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    checkpoint_dir = (
        Path("jax_impl/checkpoints")
        / args.run_name
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(args).copy()
    config["parameter_count"] = param_count

    with open(
        checkpoint_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    np.savez(
        checkpoint_dir
        / "normalization.npz",
        mean=np.asarray(norm["mean"]),
        std=np.asarray(norm["std"]),
    )

    def save_checkpoint(step_number):
        print()
        print(
            f"Saving checkpoint at step "
            f"{step_number:,}..."
        )

        jax.block_until_ready(
            state.params
        )

        payload = {
            "state": state,
            "constants": constants,
        }

        data = serialization.to_bytes(
            payload
        )

        path = (
            checkpoint_dir
            / f"step_{step_number:07d}.msgpack"
        )

        with open(path, "wb") as f:
            f.write(data)

        print("Saved:", path)

    # --------------------------------------------------------
    # Shuffle iterator
    # --------------------------------------------------------

    numpy_rng = np.random.RandomState(
        args.seed
    )

    order = numpy_rng.permutation(
        num_samples
    )

    position = 0

    def next_batch():
        nonlocal order, position

        if position >= num_samples:
            order = numpy_rng.permutation(
                num_samples
            )
            position = 0

        end = min(
            position + args.batch_size,
            num_samples,
        )

        idx = order[position:end]

        position = end

        x = np.asarray(
            windows[idx],
            dtype=np.float32,
        )

        c = np.asarray(
            conditions[idx],
            dtype=np.float32,
        )

        return (
            jax.device_put(x),
            jax.device_put(c),
        )

    # --------------------------------------------------------
    # First step = compilation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("JIT COMPILATION")
    print("=" * 70)

    batch, cond = next_batch()

    compile_start = time.time()

    state, metrics = train_step(
        state,
        batch,
        cond,
    )

    jax.block_until_ready(
        metrics["loss"]
    )

    compile_seconds = (
        time.time()
        - compile_start
    )

    print(
        "Compilation + first update:",
        format_time(
            compile_seconds
        ),
    )

    print(
        "Initial loss:",
        float(metrics["loss"]),
    )

    # --------------------------------------------------------
    # Timed training
    # --------------------------------------------------------

    completed = 1
    reported = 1

    train_start = time.time()

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()

    pbar = tqdm(
        total=args.steps,
        initial=1,
        desc="GTA Diffusion",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    while completed < args.steps:
        batch, cond = next_batch()

        state, metrics = train_step(
            state,
            batch,
            cond,
        )

        completed += 1

        checkpoint_due = (
            completed
            % args.checkpoint_every
            == 0
        )

        log_due = (
            completed
            % args.log_every
            == 0
        )

        final_step = (
            completed
            == args.steps
        )

        if (
            log_due
            or checkpoint_due
            or final_step
        ):
            jax.block_until_ready(
                metrics["loss"]
            )

            now = time.time()

            elapsed = (
                now - train_start
            )

            trained_steps = max(
                1,
                completed - 1,
            )

            steps_per_sec = (
                trained_steps
                / elapsed
            )

            remaining = (
                args.steps
                - completed
            )

            eta = (
                remaining
                / max(
                    steps_per_sec,
                    1e-9,
                )
            )

            delta = (
                completed
                - reported
            )

            if delta > 0:
                pbar.update(delta)
                reported = completed

            pbar.set_postfix(
                loss=f"{float(metrics['loss']):.4f}",
                lr=f"{float(metrics['lr']):.2e}",
                grad=f"{float(metrics['grad_norm']):.2f}",
                sps=f"{steps_per_sec:.2f}",
                elapsed=format_time(elapsed),
                ETA=format_time(eta),
            )

        if checkpoint_due:
            save_checkpoint(
                completed
            )

    pbar.close()

    total_runtime = (
        time.time()
        - train_start
    )

    # Always ensure final checkpoint exists.
    if (
        args.steps
        % args.checkpoint_every
        != 0
    ):
        save_checkpoint(
            args.steps
        )

    results_dir = Path(
        "jax_impl/outputs"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "run_name": args.run_name,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "parameter_count": param_count,
        "runtime_seconds": total_runtime,
        "runtime": format_time(
            total_runtime
        ),
        "final_loss": float(
            metrics["loss"]
        ),
        "device": str(
            jax.devices()[0]
        ),
    }

    with open(
        results_dir
        / f"{args.run_name}.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DIFFUSION TRAINING COMPLETE")
    print("=" * 70)
    print(
        "Runtime    :",
        format_time(total_runtime),
    )
    print(
        "Final loss :",
        float(metrics["loss"]),
    )
    print(
        "Checkpoint :",
        checkpoint_dir,
    )


if __name__ == "__main__":
    main()

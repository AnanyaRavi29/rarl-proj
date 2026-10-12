import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from flax import serialization
from tqdm import tqdm

from gta_jax.data import load_qlearning_dataset
from gta_jax.rl.td3bc import build_td3bc


def format_time(seconds):
    seconds = max(0, int(seconds))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


def save_actor_npz(state, path):
    params = jax.device_get(state.actor_params)

    np.savez(
        path,
        dense0_kernel=np.asarray(
            params["Dense_0"]["kernel"]
        ),
        dense0_bias=np.asarray(
            params["Dense_0"]["bias"]
        ),
        dense1_kernel=np.asarray(
            params["Dense_1"]["kernel"]
        ),
        dense1_bias=np.asarray(
            params["Dense_1"]["bias"]
        ),
        dense2_kernel=np.asarray(
            params["Dense_2"]["kernel"]
        ),
        dense2_bias=np.asarray(
            params["Dense_2"]["bias"]
        ),
    )


def save_checkpoint(state, directory, step):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        directory
        / f"step_{step:07d}.msgpack"
    )

    checkpoint_path.write_bytes(
        serialization.to_bytes(state)
    )

    actor_path = (
        directory
        / f"actor_step_{step:07d}.npz"
    )

    save_actor_npz(
        state,
        actor_path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/halfcheetah-medium-v2-qlearning.npz",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--run-name",
        default="td3bc_halfcheetah_medium_seed0",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("JAX TD3+BC")
    print("=" * 70)

    print("JAX version :", jax.__version__)
    print("Backend     :", jax.default_backend())
    print("Devices     :", jax.devices())

    buffer, stats = load_qlearning_dataset(
        args.dataset
    )

    state_dim = int(stats["state_dim"])
    action_dim = int(stats["action_dim"])

    print()
    print("Transitions :", stats["num_transitions"])
    print("State dim   :", state_dim)
    print("Action dim  :", action_dim)
    print("Batch size  :", args.batch_size)
    print("Total steps :", args.steps)
    print()

    output_dir = (
        Path("jax_impl/checkpoints")
        / args.run_name
    )

    results_dir = Path("jax_impl/outputs")
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_dir / "normalization.npz",
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
    )

    config = vars(args).copy()

    with open(
        output_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    state, run_chunk = build_td3bc(
        state_dim=state_dim,
        action_dim=action_dim,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    # ---------------------------------
    # First chunk: includes compilation
    # ---------------------------------

    first_chunk = min(
        args.chunk_size,
        args.steps,
    )

    print(
        f"Compiling JAX training loop "
        f"and running first {first_chunk} updates..."
    )

    compile_start = perf_counter()

    state, metrics = run_chunk(
        state,
        buffer,
        num_steps=first_chunk,
    )

    jax.block_until_ready(
        metrics["critic_loss"]
    )

    compile_elapsed = (
        perf_counter()
        - compile_start
    )

    completed = first_chunk

    print(
        "Compilation + first chunk: "
        f"{format_time(compile_elapsed)}"
    )

    print()
    print("Training...")
    print()

    timed_start = perf_counter()
    timed_start_step = completed

    last_checkpoint = 0

    pbar = tqdm(
        total=args.steps,
        initial=completed,
        dynamic_ncols=True,
        desc="TD3+BC",
    )

    while completed < args.steps:
        remaining = (
            args.steps
            - completed
        )

        chunk = min(
            args.chunk_size,
            remaining,
        )

        state, metrics = run_chunk(
            state,
            buffer,
            num_steps=chunk,
        )

        jax.block_until_ready(
            metrics["critic_loss"]
        )

        completed += chunk
        pbar.update(chunk)

        now = perf_counter()

        elapsed = (
            now
            - timed_start
        )

        timed_updates = (
            completed
            - timed_start_step
        )

        rate = (
            timed_updates / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining_updates = (
            args.steps
            - completed
        )

        eta = (
            remaining_updates / rate
            if rate > 0
            else 0.0
        )

        critic_loss = float(
            metrics["critic_loss"]
        )

        actor_loss = float(
            metrics["actor_loss"]
        )

        q1 = float(
            metrics["q1"]
        )

        pbar.set_postfix(
            critic=f"{critic_loss:.4f}",
            actor=f"{actor_loss:.4f}",
            q=f"{q1:.3f}",
            ups=f"{rate:.0f}",
            elapsed=format_time(elapsed),
            ETA=format_time(eta),
        )

        should_save = (
            completed
            // args.checkpoint_every
            >
            last_checkpoint
            // args.checkpoint_every
        )

        if should_save:
            save_checkpoint(
                state,
                output_dir,
                completed,
            )

            last_checkpoint = completed

    pbar.close()

    total_runtime = (
        perf_counter()
        - timed_start
    )

    save_checkpoint(
        state,
        output_dir,
        completed,
    )

    result = {
        "run_name": args.run_name,
        "steps": completed,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "runtime_seconds": total_runtime,
        "runtime": format_time(
            total_runtime
        ),
        "final_critic_loss": float(
            metrics["critic_loss"]
        ),
        "final_actor_loss": float(
            metrics["actor_loss"]
        ),
        "final_q1": float(
            metrics["q1"]
        ),
        "device": str(
            jax.devices()[0]
        ),
    }

    result_path = (
        results_dir
        / f"{args.run_name}.json"
    )

    with open(
        result_path,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Runtime    :",
        format_time(total_runtime),
    )

    print(
        "Checkpoint :",
        output_dir,
    )

    print(
        "Results    :",
        result_path,
    )


if __name__ == "__main__":
    main()

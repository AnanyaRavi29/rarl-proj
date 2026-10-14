"""Experiment A with the JAX TD3+BC (jax_impl/train_td3bc.py) instead of the PyTorch one.

Stages that need gym/MuJoCo run in the `gta` env; training runs in the `gta-jax` env:
    python scripts/h2_rl_jax.py data --conditions none uniform_0.5 ...     # gta env
    python jax_impl/train_td3bc.py --dataset ... --run-name ...            # gta-jax env (see h2_rl_jax_all.sh)
    python scripts/h2_rl_jax.py eval                                        # gta env
    python scripts/h2_rl.py report --out outputs/h2_rl_jax                  # same report/bar graph
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

ENV = "halfcheetah-medium-v2"
CKPT_ROOT = Path("jax_impl/checkpoints")


def cmd_data(args):
    """Flat (s, a, r, s', done) file per condition: original 1M transitions + that condition's GTA data."""
    from src.h2.rl_train import load_dataset
    out = Path(args.out) / "jax_data"
    out.mkdir(parents=True, exist_ok=True)
    for cond in args.conditions:
        path = out / f"{cond}.npz"
        if path.exists():
            print(f"{cond}: exists")
            continue
        gta = None if cond == "none" else Path(args.gta_out) / "data" / f"{cond}.npz"
        d = load_dataset(ENV, gta)
        np.savez(path, **d)
        print(f"{cond}: {len(d['observations'])} transitions -> {path}")


def _relu(x):
    return np.maximum(x, 0.0)


def _evaluate(run_dir, steps, episodes, seed):
    import gym
    import d4rl  # noqa: F401
    env = gym.make(ENV)
    norm = np.load(run_dir / "normalization.npz")
    mean, std = norm["state_mean"], norm["state_std"]
    rows = []
    for step in steps:
        p = np.load(run_dir / f"actor_step_{step:07d}.npz")
        returns = []
        for ep in range(episodes):
            env.seed(seed + ep)
            s, done, ret = env.reset(), False, 0.0
            while not done:
                x = (s - mean) / std
                x = _relu(x @ p["dense0_kernel"] + p["dense0_bias"])
                x = _relu(x @ p["dense1_kernel"] + p["dense1_bias"])
                a = np.tanh(x @ p["dense2_kernel"] + p["dense2_bias"])
                s, r, done, _ = env.step(np.clip(a, -1, 1))
                ret += r
            returns.append(ret)
        rows.append((step, float(env.get_normalized_score(np.mean(returns)) * 100)))
    return rows


def _eval_run(task):
    run_name, out, episodes, seed = task
    run_dir = CKPT_ROOT / run_name
    cond, s = run_name[len("h2rl_"):].rsplit("_seed", 1)
    dest = Path(out) / "runs" / f"{cond}_seed{s}"
    if (dest / "done.json").exists():
        return f"{dest.name}: already evaluated"
    steps = sorted(set(list(range(50_000, 1_000_001, 50_000)) + [960_000, 970_000, 980_000, 990_000]))
    steps = [st for st in steps if (run_dir / f"actor_step_{st:07d}.npz").exists()]
    if 1_000_000 not in steps:
        return f"{run_name}: not finished, skipped"
    rows = _evaluate(run_dir, steps, episodes, seed)
    dest.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows, columns=["step", "d4rl_score"]).to_csv(dest / "evals.csv", index=False)
    final = float(np.mean([v for st, v in rows if st >= 960_000]))
    (dest / "done.json").write_text(json.dumps({"condition": cond, "seed": int(s), "final_score_last5": final,
                                                "implementation": "jax"}, indent=2))
    for f in run_dir.glob("step_*.msgpack"):  # full training states are large; actors are kept
        if not f.name.endswith("1000000.msgpack"):
            f.unlink()
    return f"{dest.name}: final D4RL score (last 5 evals) = {final:.2f}"


def cmd_eval(args):
    runs = sorted(p.name for p in CKPT_ROOT.glob("h2rl_*_seed*") if p.is_dir())
    tasks = [(r, args.out, args.episodes, 0) for r in runs]
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(_eval_run, tasks):
            print(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["data", "eval"])
    p.add_argument("--out", default="outputs/h2_rl_jax")
    p.add_argument("--gta_out", default="outputs/h2_rl", help="where the GTA datasets were generated")
    p.add_argument("--conditions", nargs="+", default=["none", "uniform_0.3", "adaptive", "inverted", "uniform_0.5",
                                                       "adaptive_wide", "inverted_wide"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    {"data": cmd_data, "eval": cmd_eval}[args.stage](args)


if __name__ == "__main__":
    main()

"""Quick pilot: does quality-based mu help TD3+BC? (mu by quality quintile, ranked within support quintiles)

    python scripts/h2_quality_pilot.py gen    # gta env: subset of sources, 3 GTA datasets + JAX data files
    python scripts/h2_quality_pilot.py eval   # gta env: evaluate JAX actors, bar graph
"""
import argparse, json
from pathlib import Path
import numpy as np
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h2_quality_pilot")
CONDS = ["uniform_0.5", "quality_adaptive", "quality_inverted"]
ENV = "halfcheetah-medium-v2"


def quality_bins(src):
    """Quality quintile computed separately inside each support quintile (removes the support confound)."""
    q_bin = np.empty(len(src["q_return"]), int)
    for b in range(5):
        ix = np.flatnonzero(src["s_bin"] == b)
        pct = (np.argsort(np.argsort(src["q_return"][ix])) + 0.5) / len(ix)
        q_bin[ix] = np.minimum((pct * 5).astype(int), 4)
    return q_bin


def cmd_gen(args):
    import gym, d4rl  # noqa
    from src.h2.common import WindowData, build_model, load_cfg, pick_device
    from src.h2.rl_data import generate_condition
    from src.h2.rl_train import load_dataset
    ckpt = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"
    cfg = load_cfg(ckpt)
    wd = WindowData(ENV, cfg.Dataset.seq_len, cfg.Dataset.gamma)
    device = pick_device("auto")
    model = build_model(ckpt, cfg, wd, device)
    full = dict(np.load("outputs/h2_rl/data/sources.npz"))
    full["q_bin"] = quality_bins(full)
    ix = np.sort(np.random.default_rng(1).choice(len(full["episode"]), args.n_windows, replace=False))
    src = {k: v[ix] for k, v in full.items()}
    src["s_bin"] = src["q_bin"]  # generate_condition indexes the mu table with this column
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "jax_data").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "data" / "sources_subset.npz", **src)
    env = gym.make(ENV)
    for c in CONDS:
        generate_condition(c, src, wd, model, env, cfg, 2048, device, 0, OUT / "data", n_check=500)
        np.savez(OUT / "jax_data" / f"{c}.npz", **load_dataset(ENV, OUT / "data" / f"{c}.npz"))
        print(f"{c}: JAX data ready")


def cmd_eval(args):
    import pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    from scripts.h2_rl_jax import _evaluate  # noqa
    rows = []
    for c in CONDS:
        for s in args.seeds:
            d = Path("jax_impl/checkpoints") / f"qpilot_{c}_seed{s}"
            steps = [args.steps - 10_000 * k for k in range(5)]
            sc = _evaluate(d, steps, 10, 0)
            rows.append(dict(condition=c, seed=s, final=float(np.mean([v for _, v in sc]))))
            print(c, s, round(rows[-1]["final"], 2))
    df = pd.DataFrame(rows); df.to_csv(OUT / "results.csv", index=False)
    t = df.groupby("condition").final.agg(["mean", "std"]).reindex(CONDS)
    print(t.round(2))
    a = df[df.condition == "quality_adaptive"].final
    for c in ["uniform_0.5", "quality_inverted"]:
        b = df[df.condition == c].final
        print(f"quality_adaptive vs {c}: {a.mean()-b.mean():+.2f}  (Welch p={ttest_ind(a, b, equal_var=False).pvalue:.3f})")
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(CONDS):
        v = df[df.condition == c].final
        ax.bar(i, v.mean(), color=["tab:blue", "tab:red", "tab:orange"][i], alpha=0.6)
        ax.scatter([i] * len(v), v, color="k", s=15, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(CONDS)
    ax.set_ylim(df.final.min() - 3, df.final.max() + 1)
    ax.set_ylabel("D4RL score (mean of last 5 evals)")
    ax.set_title(f"PILOT: mu by quality ({args.steps // 1000}k steps, {len(args.seeds)} seeds)")
    fig.tight_layout(); fig.savefig(OUT / "pilot_bars.png", dpi=120)
    print(f"Saved {OUT / 'pilot_bars.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["gen", "eval"])
    p.add_argument("--n_windows", type=int, default=9700)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = p.parse_args()
    {"gen": cmd_gen, "eval": cmd_eval}[args.stage](args)

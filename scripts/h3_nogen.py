"""H3: is the diffusion model necessary? No-generation baselines vs GTA (JAX TD3+BC).

Each condition = original 1M transitions + 5M extra transitions obtained by RESAMPLING the original data
(no diffusion model), mirroring GTA's "original + generated" structure:
  rw_b{beta}  sample transitions with probability proportional to exp(beta * z(q)), z = standardised quality
  top{k}      copies of the top-k% transitions by quality
where quality q_t = discounted reward over the next 31 steps (the GTA window), within the episode.
Reference: gta_ref = existing GTA data (mu 0.5, alpha 1.3, 1M generated), read only.

    python scripts/h3_nogen.py data      # gta env
    python scripts/h3_nogen.py eval      # gta env
    python scripts/h3_nogen.py report
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h3_nogen")
ENV = "halfcheetah-medium-v2"
CKPT = Path("jax_impl/checkpoints")
N_EXTRA = 5_000_000
GTA_REF = Path("outputs/h2_alpha/jax_data/alpha_1.3.npz")
CONDS = ["none", "rw_b0.5", "rw_b1", "rw_b2", "top25", "top10", "gta_ref"]


def transition_quality(gamma=0.99, H=31):
    """Per-transition discounted reward over the next H steps (same horizon as a GTA window)."""
    import pickle
    with open(f"data/{ENV}.pkl", "rb") as f:
        eps = pickle.load(f)
    kernel = gamma ** np.arange(H)
    qs = []
    for ep in eps:
        r = np.asarray(ep["rewards"], np.float64).reshape(-1)
        padded = np.concatenate([r, np.zeros(H - 1)])
        qs.append(np.convolve(padded, kernel[::-1], mode="valid")[:len(r)])
    return np.concatenate(qs)


def cmd_data(args):
    from src.h2.rl_train import load_dataset
    (OUT / "jax_data").mkdir(parents=True, exist_ok=True)
    orig = load_dataset(ENV, None)
    q = transition_quality()
    assert len(q) == len(orig["observations"])
    z = (q - q.mean()) / q.std()
    rng = np.random.default_rng(0)
    for c in CONDS:
        path = OUT / "jax_data" / f"{c}.npz"
        if c == "gta_ref" or path.exists():
            continue
        if c == "none":
            extra = None
        elif c.startswith("rw_b"):
            w = np.exp(float(c[4:]) * z)
            extra = rng.choice(len(z), N_EXTRA, replace=True, p=w / w.sum())
        else:
            k = int(c[3:])
            top = np.flatnonzero(q >= np.percentile(q, 100 - k))
            extra = rng.choice(top, N_EXTRA, replace=True)
        d = orig if extra is None else {key: np.concatenate([v, v[extra]]) for key, v in orig.items()}
        np.savez(path, **d)
        info = "original only" if extra is None else (
            f"extra: mean quality {q[extra].mean():.1f} (original {q.mean():.1f}), unique {len(np.unique(extra)):,}")
        print(f"{c}: {len(d['observations']):,} transitions; {info}", flush=True)
    if not (OUT / "jax_data" / "gta_ref.npz").exists():
        os.symlink(GTA_REF.resolve(), OUT / "jax_data" / "gta_ref.npz")


def _eval_one(run):
    os.environ["OMP_NUM_THREADS"] = "1"
    from scripts.h2_rl_jax import _evaluate
    rows = _evaluate(CKPT / run, [960_000, 970_000, 980_000, 990_000, 1_000_000], 10, 0)
    return run, float(np.mean([v for _, v in rows]))


def cmd_eval(args):
    runs = sorted(p.name for p in CKPT.glob("h3__*") if (p / "actor_step_1000000.npz").exists())
    res_f = OUT / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    with ProcessPoolExecutor(10) as ex:
        for run, score in ex.map(_eval_one, [r for r in runs if r not in res]):
            res[run] = score
            print(f"{run}: {score:.2f}", flush=True)
    res_f.write_text(json.dumps(res, indent=1))


def cmd_report(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    res = json.loads((OUT / "results.json").read_text())
    df = pd.DataFrame([dict(condition=k[4:].rsplit("_seed", 1)[0], seed=int(k.rsplit("_seed", 1)[1]), final=v)
                       for k, v in res.items()])
    t = df.groupby("condition").final.agg(["mean", "std", "count"]).reindex([c for c in CONDS if c in set(df.condition)])
    lines = ["# H3: is the diffusion model necessary? (JAX TD3+BC, 1M steps, 3 seeds)", "",
             "Extra data: 5M transitions resampled from the original data (no generation); gta_ref = GTA mu 0.5, alpha 1.3 (1M generated).", "",
             "| condition | mean | std | seeds |", "|---|---|---|---|"]
    lines += [f"| {c} | {r['mean']:.2f} | {r['std']:.2f} | {int(r['count'])} |" for c, r in t.iterrows()]
    g, n = df[df.condition == "gta_ref"].final, df[df.condition == "none"].final
    lines.append("")
    for c in t.index:
        if c in ("gta_ref", "none"):
            continue
        v = df[df.condition == c].final
        if len(v) >= 2 and len(g) >= 2 and len(n) >= 2:
            gain = (v.mean() - n.mean()) / (g.mean() - n.mean()) * 100
            lines.append(f"- {c}: vs none {v.mean() - n.mean():+.2f} (p = {ttest_ind(v, n, equal_var=False).pvalue:.3f}), "
                         f"vs GTA {v.mean() - g.mean():+.2f} (p = {ttest_ind(v, g, equal_var=False).pvalue:.3f}); "
                         f"recovers {gain:.0f}% of GTA's gain")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, c in enumerate(t.index):
        v = df[df.condition == c].final
        ax.bar(i, v.mean(), color="tab:red" if c == "gta_ref" else "grey" if c == "none" else "tab:blue", alpha=0.6)
        ax.scatter([i] * len(v), v, color="k", s=15, zorder=3)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t.index, rotation=15)
    ax.set_ylim(df.final.min() - 3, df.final.max() + 1)
    ax.set_ylabel("D4RL score (mean of last 5 evals)")
    ax.set_title("No-generation resampling vs GTA (halfcheetah-medium)")
    fig.tight_layout()
    fig.savefig(OUT / "h3_bars.png", dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["data", "eval", "report"])
    a = p.parse_args()
    {"data": cmd_data, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)

"""Alpha experiment (return multiplier of GTA's guidance), all at mu = 0.5, 1M generated transitions.

Stage 1: one global alpha in {1.0, 1.1, 1.3, 1.5, 1.8, 2.0}.
Stage 2: alpha by support quintile around the best global alpha a*: up = a*-0.2 .. a*+0.2, down = reverse.

    python scripts/h2_alpha.py gen --stage 1        # gta env: datasets + JAX data files
    python scripts/h2_alpha.py eval                 # gta env: evaluate all trained runs
    python scripts/h2_alpha.py report               # gta env: bars, stats, best alpha (stage1.json)
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h2_alpha")
ENV = "halfcheetah-medium-v2"
CKPT = Path("jax_impl/checkpoints")
ALPHAS = [1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
N_WINDOWS = 32_258  # x31 = 1,000,000 generated transitions


def stage_conditions(stage, src):
    """name -> per-window alpha array (mu is 0.5 everywhere)."""
    n = len(src["s_bin"])
    if stage == 1:
        return {f"alpha_{a:g}": np.full(n, a, np.float32) for a in ALPHAS}
    if stage == 3:  # same alpha values as alpha_up/down, assigned at random (ignores support)
        vals = np.array([1.1, 1.2, 1.3, 1.4, 1.5], np.float32)
        return {"alpha_random": vals[np.random.default_rng(3).integers(0, 5, n)]}
    best = json.loads((OUT / "stage1.json").read_text())["best_alpha"]
    steps = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], np.float32)
    up = np.maximum(best + steps, 1.0)
    return {"alpha_up": up[src["s_bin"]], "alpha_down": up[::-1][src["s_bin"]]}


def cmd_gen(args):
    import gym, d4rl  # noqa: F401
    from src.h2.common import WindowData, build_model, load_cfg, pick_device
    from src.h2.rl_data import generate_condition
    from src.h2.rl_train import load_dataset
    ckpt = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"
    cfg = load_cfg(ckpt)
    wd = WindowData(ENV, cfg.Dataset.seq_len, cfg.Dataset.gamma)
    device = pick_device("auto")
    model = build_model(ckpt, cfg, wd, device)
    full = dict(np.load("outputs/h2_rl/data/sources.npz"))  # 161,291 windows with S2 support quintile
    ix = np.sort(np.random.default_rng(2).choice(len(full["episode"]), N_WINDOWS, replace=False))
    src = {k: v[ix] for k, v in full.items()}
    for d in ["data", "jax_data"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV)
    for name, alphas in stage_conditions(args.stage, src).items():
        jpath = OUT / "jax_data" / f"{name}.npz"
        if jpath.exists():
            continue
        generate_condition(name, src, wd, model, env, cfg, 2048, device, 0, OUT / "data", n_check=1000,
                           mus=np.full(len(alphas), 0.5, np.float32), alphas=alphas)
        np.savez(jpath, **load_dataset(ENV, OUT / "data" / f"{name}.npz"))
        print(f"{name}: JAX data ready (mean alpha {alphas.mean():.3f})", flush=True)


def _eval_one(run):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from scripts.h2_rl_jax import _evaluate
    d = CKPT / run
    steps = [960_000, 970_000, 980_000, 990_000, 1_000_000]
    curve = list(range(100_000, 1_000_001, 100_000))
    rows = _evaluate(d, sorted(set(steps + curve)), 10, 0)
    final = float(np.mean([v for s, v in rows if s in steps]))
    return run, final, rows


def cmd_eval(args):
    runs = sorted(p.name for p in CKPT.glob("alpha__*") if (p / "actor_step_1000000.npz").exists())
    res = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    todo = [r for r in runs if r not in res]
    with ProcessPoolExecutor(min(12, max(1, len(todo)))) as ex:
        for run, final, rows in ex.map(_eval_one, todo):
            res[run] = {"final": final, "curve": rows}
            print(f"{run}: {final:.2f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


STAGE3 = ["alpha_1.3", "alpha_down", "alpha_random", "alpha_up"]


def stage3_report(df, ttest_ind, plt):
    """Fresh seeds 10-14: does matching alpha to support beat a random alpha mix with the same values?"""
    d = df[(df.seed >= 10) & df.condition.isin(STAGE3)]
    if not len(d):
        return []
    t = d.groupby("condition").final.agg(["mean", "std", "count"]).reindex(STAGE3)
    out = ["", "## Stage 3: support vs random alpha mix (fresh seeds 10-14, mean alpha 1.3)", "",
           "| condition | mean | std | seeds |", "|---|---|---|---|"]
    out += [f"| {c} | {r['mean']:.2f} | {r['std']:.2f} | {int(r['count'])} |" for c, r in t.iterrows() if r['count'] > 0]
    def test(a, b):
        x, y = d[d.condition == a].final, d[d.condition == b].final
        if len(x) >= 2 and len(y) >= 2:
            out.append(f"- {a} vs {b}: {x.mean() - y.mean():+.2f} (Welch p = {ttest_ind(x, y, equal_var=False).pvalue:.3f})")
    for a, b in [("alpha_up", "alpha_random"), ("alpha_up", "alpha_down"), ("alpha_random", "alpha_1.3"), ("alpha_up", "alpha_1.3")]:
        test(a, b)
    out += ["", "Reading: alpha_up > alpha_random means support matters; alpha_random > alpha_1.3 with alpha_up ~ alpha_random means mixing alpha matters."]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = {"alpha_1.3": "tab:blue", "alpha_down": "tab:orange", "alpha_random": "tab:green", "alpha_up": "tab:red"}
    for i, c in enumerate(STAGE3):
        v = d[d.condition == c].final
        if len(v):
            ax.bar(i, v.mean(), color=cols[c], alpha=0.6)
            ax.scatter([i] * len(v), v, color="k", s=15, zorder=3)
    ax.set_xticks(range(len(STAGE3)))
    ax.set_xticklabels(["uniform 1.3", "down (1.5 to 1.1)", "random mix", "up (1.1 to 1.5)"])
    ax.set_ylim(d.final.min() - 2, d.final.max() + 1)
    ax.set_ylabel("D4RL score (mean of last 5 evals)")
    ax.set_title("Alpha by support vs random alpha mix (5 fresh seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "alpha_stage3_bars.png", dpi=120)
    return out


def cmd_report(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    res = json.loads((OUT / "results.json").read_text())
    df = pd.DataFrame([dict(condition=k[len("alpha__"):].rsplit("_seed", 1)[0], seed=int(k.rsplit("_seed", 1)[1]),
                            final=v["final"]) for k, v in res.items()])
    order = [f"alpha_{a:g}" for a in ALPHAS] + ["alpha_up", "alpha_down"]
    t = df.groupby("condition").final.agg(["mean", "std", "count"]).reindex([c for c in order if c in set(df.condition)])
    lines = ["# Alpha experiment (mu = 0.5, 1M generated transitions, JAX TD3+BC, 3 seeds)", "",
             "| condition | mean | std | seeds |", "|---|---|---|---|"]
    lines += [f"| {c} | {r['mean']:.2f} | {r['std']:.2f} | {int(r['count'])} |" for c, r in t.iterrows()]
    s1 = t.loc[[c for c in t.index if c.startswith("alpha_") and c[6:].replace(".", "").isdigit()]]
    if len(s1):
        best = float(s1["mean"].idxmax()[6:])
        (OUT / "stage1.json").write_text(json.dumps({"best_alpha": best, "scores": s1["mean"].round(3).to_dict()}))
        lines += ["", f"**Stage 1 best global alpha: {best:g}**"]
        b = df[df.condition == f"alpha_{best:g}"].final
        for c in ["alpha_up", "alpha_down"]:
            a = df[df.condition == c].final
            if len(a) >= 2:
                lines.append(f"- {c} vs alpha_{best:g}: {a.mean() - b.mean():+.2f} (Welch p = {ttest_ind(a, b, equal_var=False).pvalue:.3f})")
        u, d = df[df.condition == "alpha_up"].final, df[df.condition == "alpha_down"].final
        if len(u) >= 2 and len(d) >= 2:
            lines.append(f"- alpha_up vs alpha_down: {u.mean() - d.mean():+.2f} (Welch p = {ttest_ind(u, d, equal_var=False).pvalue:.3f})")
    lines += stage3_report(df, ttest_ind, plt)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, c in enumerate(t.index):
        v = df[df.condition == c].final
        ax.bar(i, v.mean(), color="tab:red" if c == "alpha_up" else "tab:orange" if c == "alpha_down" else "tab:blue", alpha=0.6)
        ax.scatter([i] * len(v), v, color="k", s=15, zorder=3)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t.index, rotation=15)
    ax.set_ylim(df.final.min() - 3, df.final.max() + 1)
    ax.set_ylabel("D4RL score (mean of last 5 evals)")
    ax.set_title("Return multiplier alpha (mu = 0.5): global sweep and support-adaptive")
    fig.tight_layout()
    fig.savefig(OUT / "alpha_bars.png", dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage_cmd", choices=["gen", "eval", "report"])
    p.add_argument("--stage", type=int, default=1)
    args = p.parse_args()
    {"gen": cmd_gen, "eval": cmd_eval, "report": cmd_report}[args.stage_cmd](args)

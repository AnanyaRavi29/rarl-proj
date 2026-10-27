"""TWGA Gate 0 (oracle diagnostic): can down-weighting physically implausible generated transitions raise
GTA's alpha ceiling? Weights come from the TRUE simulator error (diagnostic only, not the method):
    w = 1 for original transitions,  w = exp(-e / tau) for generated ones,
    e = mean over the 17 state dims of ((f*(s, a) - s') / sd)^2.
Uses the existing GTA datasets alpha 1.5 / 1.8 / 2.0 (mu 0.5, 1M generated) from outputs/h2_alpha (read only).

    python scripts/h6_gate0.py errors     # simulator error of every generated transition (CPU, parallel)
    python scripts/h6_gate0.py data       # weighted JAX data files
    python scripts/h6_gate0.py eval / report
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h6_twga/gate0")
ENV = "halfcheetah-medium-v2"
ALPHAS = ["1.5", "1.8", "2"]
TAU = 0.0085
CKPT = Path("jax_impl/checkpoints")


def _state_std():
    import pickle
    with open(f"data/{ENV}.pkl", "rb") as f:
        eps = pickle.load(f)
    return np.concatenate([e["observations"] for e in eps]).std(0) + 1e-6


def _errors_chunk(task):
    path, lo, hi, std = task
    os.environ["OMP_NUM_THREADS"] = "1"
    import gym
    import d4rl  # noqa: F401
    z = np.load(path)
    s, a, s2 = z["observations"][lo:hi], z["actions"][lo:hi], z["next_observations"][lo:hi]
    sim = gym.make(ENV).unwrapped
    nq = sim.model.nq
    out = np.empty(hi - lo, np.float32)
    for i in range(hi - lo):
        x = s[i].astype(np.float64)
        sim.set_state(np.concatenate([[0.0], x[:nq - 1]]), x[nq - 1:])
        nxt, _, _, _ = sim.step(a[i])
        out[i] = np.mean(((nxt - s2[i]) / std) ** 2)
    return lo, out


def cmd_errors(args):
    OUT.mkdir(parents=True, exist_ok=True)
    std = _state_std()
    for a in ALPHAS:
        f = OUT / f"err_alpha_{a}.npy"
        if f.exists():
            continue
        path = f"outputs/h2_alpha/data/alpha_{a}.npz"
        n = len(np.load(path)["rewards"])
        err = np.empty(n, np.float32)
        tasks = [(path, lo, min(lo + 50_000, n), std) for lo in range(0, n, 50_000)]
        with ProcessPoolExecutor(args.workers) as ex:
            for lo, e in ex.map(_errors_chunk, tasks):
                err[lo:lo + len(e)] = e
        np.save(f, err)
        w = np.exp(-err / TAU)
        print(f"alpha {a}: {n:,} generated transitions | error mean {err.mean():.4f}, median {np.median(err):.4f} | "
              f"share > {TAU}: {(err > TAU).mean():.2f} | mean weight {w.mean():.3f} "
              f"(effective generated data {w.sum():,.0f})", flush=True)


def cmd_data(args):
    from src.h2.rl_train import load_dataset
    (OUT / "jax_data").mkdir(parents=True, exist_ok=True)
    for a in ALPHAS:
        dst = OUT / "jax_data" / f"w_alpha_{a}.npz"
        if dst.exists():
            continue
        d = load_dataset(ENV, f"outputs/h2_alpha/data/alpha_{a}.npz")  # original 1M first, then generated
        err = np.load(OUT / f"err_alpha_{a}.npy")
        n_orig = len(d["observations"]) - len(err)
        d["weights"] = np.concatenate([np.ones(n_orig, np.float32), np.exp(-err / TAU).astype(np.float32)])
        np.savez(dst, **d)
        print(f"w_alpha_{a}: {len(d['weights']):,} transitions ({n_orig:,} original)", flush=True)


def _eval_one(run):
    os.environ["OMP_NUM_THREADS"] = "1"
    from scripts.h2_rl_jax import _evaluate
    rows = _evaluate(CKPT / run, [960_000, 970_000, 980_000, 990_000, 1_000_000], 10, 0)
    return run, float(np.mean([v for _, v in rows]))


def cmd_eval(args):
    runs = sorted(p.name for p in CKPT.glob("h6g0__*") if (p / "actor_step_1000000.npz").exists())
    f = OUT / "results.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    with ProcessPoolExecutor(9) as ex:
        for run, sc in ex.map(_eval_one, [r for r in runs if r not in res]):
            res[run] = sc
            print(f"{run}: {sc:.2f}", flush=True)
    f.write_text(json.dumps(res, indent=1))


def cmd_report(args):
    from scipy.stats import ttest_ind
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = json.loads((OUT / "results.json").read_text())
    unw = json.loads(Path("outputs/h2_alpha/results.json").read_text())  # same data, same seeds 0-2, unweighted
    def pick(d, prefix, a):
        return [v["final"] if isinstance(v, dict) else v for k, v in d.items()
                if k.startswith(f"{prefix}alpha_{a}_seed") and int(k.rsplit("_seed", 1)[1]) in (0, 1, 2)]
    ref13 = pick(unw, "alpha__", "1.3")
    lines = ["# TWGA Gate 0: oracle trust weights (true physics error), JAX TD3+BC, seeds 0-2", "",
             f"Weight for generated transitions: exp(-error / {TAU}); original transitions weight 1.", "",
             f"Reference: unweighted GTA alpha 1.3 = {np.mean(ref13):.2f}; best tuned global (mu 0.565, alpha 1.356) = 59.05.", "",
             "| alpha | unweighted | oracle-weighted | difference | p |", "|---|---|---|---|---|"]
    xs, u_m, w_m = [], [], []
    for a in ALPHAS:
        u, v = pick(unw, "alpha__", a), pick(w, "h6g0__w_", a)
        if len(u) >= 2 and len(v) >= 2:
            p = ttest_ind(v, u, equal_var=False).pvalue
            lines.append(f"| {a} | {np.mean(u):.2f} ± {np.std(u, ddof=1):.2f} | {np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} | "
                         f"{np.mean(v) - np.mean(u):+.2f} | {p:.3f} |")
            xs.append(float(a)); u_m.append(np.mean(u)); w_m.append(np.mean(v))
    best = max(w_m) if w_m else float("nan")
    lines += ["", f"**Gate 0 {'PASSED' if best > 59.05 else 'FAILED'}:** best oracle-weighted score {best:.2f} "
              f"vs 59.05 needed (beats the best tuned global setting)."]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([1.3] + xs, [np.mean(ref13)] + u_m, "o-", label="GTA, unweighted")
    ax.plot(xs, w_m, "s-", color="tab:red", label="oracle trust-weighted")
    ax.axhline(59.05, color="k", ls="--", lw=1, label="best tuned global (59.05)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("D4RL score")
    ax.set_title("Gate 0: does trust weighting raise the alpha ceiling?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gate0.png", dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["errors", "data", "eval", "report"])
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()
    {"errors": cmd_errors, "data": cmd_data, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)

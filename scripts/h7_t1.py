"""H7 / T1: does the conservative-augmentation bound explain GTA's alpha optimum? (no RL training)

Bound (Lipschitz simulation lemma + improvement term), per generated dataset G_alpha:
    B_c(alpha) = eta(alpha) / (1 - gamma) - c * d(alpha) / (1 - gamma)
    eta(alpha) = E_G[ r_true(s, a) ] - E_D[ r ]                 (true reward gained per transition)
    d(alpha)   = E_G[ || (f*(s, a) - s') / sd ||_2 ]            (next-state error, dataset-std units)
    c          = gamma * L_V  (Lipschitz constant of V, unknown)
Compared with measured TD3+BC scores of the same datasets (outputs/h2_alpha, seeds 0-2).

    python scripts/h7_t1.py measure    # simulator replay of all generated transitions (CPU, parallel)
    python scripts/h7_t1.py analyse
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h7_conservative/t1")
ENV = "halfcheetah-medium-v2"
ALPHAS = ["1", "1.1", "1.3", "1.5", "1.8", "2"]
GAMMA = 0.99


def _chunk(task):
    path, lo, hi, std = task
    os.environ["OMP_NUM_THREADS"] = "1"
    import gym
    import d4rl  # noqa: F401
    z = np.load(path)
    s, a, s2 = z["observations"][lo:hi], z["actions"][lo:hi], z["next_observations"][lo:hi]
    sim = gym.make(ENV).unwrapped
    nq = sim.model.nq
    dist, rew = np.empty(hi - lo, np.float32), np.empty(hi - lo, np.float32)
    for i in range(hi - lo):
        x = s[i].astype(np.float64)
        sim.set_state(np.concatenate([[0.0], x[:nq - 1]]), x[nq - 1:])
        nxt, r, _, _ = sim.step(a[i])
        dist[i] = np.linalg.norm((nxt - s2[i]) / std)
        rew[i] = r
    return lo, dist, rew


def cmd_measure(args):
    import pickle
    OUT.mkdir(parents=True, exist_ok=True)
    with open(f"data/{ENV}.pkl", "rb") as f:
        eps = pickle.load(f)
    std = np.concatenate([e["observations"] for e in eps]).std(0) + 1e-6
    for a in ALPHAS:
        f = OUT / f"alpha_{a}.npz"
        if f.exists():
            continue
        path = f"outputs/h2_alpha/data/alpha_{a}.npz"
        n = len(np.load(path)["rewards"])
        dist, rew = np.empty(n, np.float32), np.empty(n, np.float32)
        with ProcessPoolExecutor(args.workers) as ex:
            for lo, d, r in ex.map(_chunk, [(path, lo, min(lo + 50_000, n), std) for lo in range(0, n, 50_000)]):
                dist[lo:lo + len(d)], rew[lo:lo + len(r)] = d, r
        gen_r = np.load(path)["rewards"]
        np.savez(f, dist=dist, true_r=rew, gen_r=gen_r)
        print(f"alpha {a}: true reward/step {rew.mean():.3f} (model {gen_r.mean():.3f}), "
              f"next-state error {dist.mean():.4f} (median {np.median(dist):.4f})", flush=True)


def cmd_analyse(args):
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    with open(f"data/{ENV}.pkl", "rb") as f:
        eps = pickle.load(f)
    r_data = np.concatenate([np.asarray(e["rewards"]).reshape(-1) for e in eps]).mean()
    res = json.loads(Path("outputs/h2_alpha/results.json").read_text())
    al = np.array([float(a) for a in ALPHAS])
    eta, d, rl = [], [], []
    for a in ALPHAS:
        z = np.load(OUT / f"alpha_{a}.npz")
        eta.append(z["true_r"].mean() - r_data)
        d.append(z["dist"].mean())
        rl.append(np.mean([v["final"] for k, v in res.items()
                           if k.startswith(f"alpha__alpha_{a}_seed") and int(k.rsplit("_seed", 1)[1]) < 3]))
    eta, d, rl = map(np.array, (eta, d, rl))
    cs = np.logspace(-2, 3, 400)
    B = (eta[None] - cs[:, None] * d[None]) / (1 - GAMMA)
    argmax = al[B.argmax(1)]
    rho = np.array([spearmanr(b, rl)[0] for b in B])
    ok = (argmax >= 1.3) & (argmax <= 1.5)
    rl_best = al[rl.argmax()]
    # theory's worst-case constant: L_V <= R_max / (1 - gamma) with R_max = max per-step reward magnitude
    r_max = np.abs(np.concatenate([np.asarray(e["rewards"]).reshape(-1) for e in eps])).max()
    c_theory = GAMMA * r_max / (1 - GAMMA)
    a_theory = al[int(np.argmax(eta - c_theory * d))]
    lines = ["# H7 / T1: does the bound explain GTA's alpha optimum?", "",
             "B_c(alpha) = [eta(alpha) - c * d(alpha)] / (1 - gamma); eta = true reward gain per generated transition, "
             "d = mean next-state error (dataset-std units), c = gamma * L_V.", "",
             "| alpha | eta (true reward gain / step) | d (next-state error) | measured RL score |", "|---|---|---|---|"]
    lines += [f"| {a} | {e:+.3f} | {x:.4f} | {s:.2f} |" for a, e, x, s in zip(ALPHAS, eta, d, rl)]
    lines += ["", f"- Measured RL optimum: alpha = {rl_best:g}.",
              f"- Worst-case theoretical constant c = gamma * R_max / (1 - gamma) = {c_theory:.0f} (R_max = {r_max:.2f}) "
              f"-> bound maximiser alpha = {a_theory:g}.",
              f"- Range of c whose maximiser lies in [1.3, 1.5]: "
              + (f"{cs[ok].min():.3g} to {cs[ok].max():.3g}" if ok.any() else "none") + ".",
              f"- Best rank agreement between B_c and the RL curve: Spearman {rho.max():.2f} at c = {cs[rho.argmax()]:.3g} "
              f"(maximiser alpha = {argmax[rho.argmax()]:g})."]
    verdict = ok.any() and rho.max() >= 0.8
    lines += ["", f"**T1 {'PASSED' if verdict else 'FAILED'}:** "
              + ("a range of c reproduces the interior optimum and the curve shape (explanatory only: c fitted on one dataset)."
                 if verdict else "the bound does not reproduce the measured alpha curve.")]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(al, eta, "o-", label="eta: true reward gain / step")
    ax2 = ax[0].twinx()
    ax2.plot(al, d, "s-", color="tab:red", label="d: next-state error")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("eta")
    ax2.set_ylabel("d", color="tab:red")
    ax[0].set_title("Improvement and model error vs alpha")
    ax[1].semilogx(cs, argmax, "-")
    ax[1].axhspan(1.3, 1.5, color="tab:green", alpha=0.2, label="measured RL optimum region")
    ax[1].set_xlabel("c = gamma * L_V")
    ax[1].set_ylabel("alpha maximising the bound")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Bound maximiser vs constant c")
    cb = cs[rho.argmax()]
    bb = (eta - cb * d) / (1 - GAMMA)
    ax[2].plot(al, (bb - bb.min()) / (np.ptp(bb) + 1e-9), "o-", label=f"bound B_c (c={cb:.3g}), scaled")
    ax[2].plot(al, (rl - rl.min()) / np.ptp(rl), "s-", label="measured RL score, scaled")
    ax[2].set_xlabel("alpha")
    ax[2].legend(fontsize=8)
    ax[2].set_title(f"Shape agreement (Spearman {rho.max():.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "t1.png", dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["measure", "analyse"])
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()
    {"measure": cmd_measure, "analyse": cmd_analyse}[a.cmd](a)

"""Repaired GTA data for TD3+BC: GTA's generated (s, a) from the alpha datasets (mu 0.5, 1M transitions), with next
state and reward replaced by the dynamics ensemble's prediction (see scripts/h9_repair_check.py).

    python scripts/h9_repair_rl.py data                # gta env: outputs/h9_repair/jax_data/repair_alpha_<a>.npz
    bash scripts/h9_repair_rl.sh                       # trains 3 conditions x 3 seeds (gta-jax env)
    python scripts/h9_repair_rl.py eval [--upto 1000000]   # gta env: D4RL scores (works on partial runs)
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h9_repair")
CKPT = Path("jax_impl/checkpoints")
ENV = "halfcheetah-medium-v2"
ALPHAS = ["1.5", "1.8", "2"]
OBS = 17


def cmd_data(args):
    import torch
    import torch.nn as nn
    from src.h2.rl_train import load_dataset

    def mlp():
        return nn.Sequential(nn.Linear(23, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                             nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 18))

    ck = torch.load("outputs/h5_vae_dyn/dyn.pt", map_location="cuda")
    ens = nn.ModuleList([mlp() for _ in range(5)]).cuda()
    ens.load_state_dict(ck["ens"])
    ens.eval()
    st = {k: v.cuda() for k, v in ck["stats"].items()}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "jax_data").mkdir(parents=True, exist_ok=True)
    for a in ALPHAS:
        d = dict(np.load(f"outputs/h2_alpha/data/alpha_{a}.npz"))
        s, act = d["observations"], d["actions"]
        nxt, rew = np.empty_like(s), np.empty(len(s), np.float32)
        with torch.no_grad():
            for i in range(0, len(s), 65536):
                x = torch.as_tensor(np.concatenate([s[i:i + 65536], act[i:i + 65536]], 1), device="cuda")
                y = torch.stack([m((x - st["in_mean"]) / st["in_std"]) for m in ens]).mean(0) * st["out_std"] + st["out_mean"]
                nxt[i:i + 65536] = (x[:, :OBS] + y[:, :OBS]).cpu().numpy()
                rew[i:i + 65536] = y[:, OBS].cpu().numpy()
        print(f"alpha {a}: mean |next-state change| {np.abs(nxt - d['next_observations']).mean():.4f}, "
              f"reward {d['rewards'].mean():.3f} -> {rew.mean():.3f}", flush=True)
        d["next_observations"], d["rewards"] = nxt, rew
        np.savez(OUT / "data" / f"repair_alpha_{a}.npz", **d)
        np.savez(OUT / "jax_data" / f"repair_alpha_{a}.npz", **load_dataset(ENV, OUT / "data" / f"repair_alpha_{a}.npz"))


def _eval_one(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from scripts.h2_rl_jax import _evaluate
    run, upto = job
    final_steps = [upto - 40_000 + 10_000 * k for k in range(5)]
    curve = list(range(100_000, upto + 1, 100_000))
    rows = _evaluate(CKPT / run, sorted(set(final_steps + curve)), 10, 0)
    return run, float(np.mean([v for s, v in rows if s in final_steps])), rows


def cmd_eval(args):
    from scipy.stats import ttest_ind
    runs = sorted(p.name for p in CKPT.glob("repair__*") if (p / f"actor_step_{args.upto:07d}.npz").exists())
    res = {}
    with ProcessPoolExecutor(min(9, max(1, len(runs)))) as ex:
        for run, final, rows in ex.map(_eval_one, [(r, args.upto) for r in runs]):
            res[run] = {"final": final, "curve": rows}
            print(f"{run}: {final:.2f}", flush=True)
    (OUT / f"results_{args.upto}.json").write_text(json.dumps(res, indent=1))
    ref = json.loads(Path("outputs/h2_alpha/results.json").read_text())
    lines = [f"# Repaired GTA data, TD3+BC (JAX), score = mean of evals at {args.upto - 40000}..{args.upto} steps", "",
             "| condition | mean | std | seeds | unrepaired (same alpha, 1M steps) |", "|---|---|---|---|---|"]
    for a in ALPHAS:
        v = np.array([r["final"] for k, r in res.items() if k.startswith(f"repair__repair_alpha_{a}_seed")])
        u = np.array([r["final"] for k, r in ref.items() if k.startswith(f"alpha__alpha_{a}_seed") and int(k.split("seed")[1]) < 3])
        if len(v):
            p = ttest_ind(v, u, equal_var=False).pvalue if len(v) > 1 and len(u) > 1 else float("nan")
            lines.append(f"| repaired alpha {a} | {v.mean():.2f} | {v.std(ddof=1) if len(v) > 1 else 0:.2f} | {len(v)} | "
                         f"{u.mean():.2f} (diff {v.mean() - u.mean():+.2f}, p = {p:.3f}) |")
    lines += ["", "References: GTA default (mu 0.5, alpha 1.3) 58.51-58.66; best tuned global 59.05; oracle-weighted alpha 1.5 59.56."]
    (OUT / f"summary_{args.upto}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    e = sub.add_parser("eval")
    e.add_argument("--upto", type=int, default=1_000_000)
    args = p.parse_args()
    {"data": cmd_data, "eval": cmd_eval}[args.cmd](args)

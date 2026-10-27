"""Acceleration edits: GTA editing (mu 0.5, return target alpha x RTG) with the first K steps of every window pinned
to the real source during denoising (inpainting). The model must then generate the speed-up inside the window,
starting from the source's real speed, instead of jumping to a higher speed at step 0.

Same 32,258 source windows as scripts/h2_alpha.py (seed 2 subset of outputs/h2_rl/data/sources.npz).

    python scripts/h12_accel.py gen --alpha 1.5     # gta env: data + JAX data + 1,000-window simulator check
    python scripts/h12_accel.py eval                # gta env: D4RL scores of finished runs + summary
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils
import torch

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h12_accel")
ENV = "halfcheetah-medium-v2"
CKPT_ROOT = Path("jax_impl/checkpoints")
CKPT = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"
N_WINDOWS = 32_258


@torch.no_grad()
def sample_pinned(model, samples, cond, mu, k_pin, temperature=1.0, num_steps=128):
    """GTA back-and-forth editing (same maths as ElucidatedDiffusion.sample_back_and_forth), except that the first
    k_pin rows are re-set to the source, noised to the current level, after every step (exact at the end)."""
    device = model.device
    x0 = model.normalizer.normalize(samples.to(device))
    sig = model.sample_schedule(num_steps)
    gam = torch.where((sig >= model.S_tmin) & (sig <= model.S_tmax), min(model.S_churn / num_steps, math.sqrt(2) - 1), 0.0)
    k0 = int(np.floor(num_steps * (1 - mu)))
    x = x0 + sig[k0] * torch.randn_like(x0)
    for k in range(k0, num_steps):
        sigma, sigma_next, gamma = sig[k].item(), sig[k + 1].item(), gam[k].item()
        sigma_hat = sigma + gamma * sigma
        x_hat = x + math.sqrt(sigma_hat ** 2 - sigma ** 2) * model.S_noise * torch.randn_like(x)
        d = model.score_fn(x_hat, sigma_hat, cond=cond)
        du = model.score_fn(x_hat, sigma_hat, cond=None)
        d = du + temperature * (d - du)
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next != 0:
            d2 = model.score_fn(x_next, sigma_next, cond=cond)
            x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d2)
        x_next[:, :k_pin] = x0[:, :k_pin] + sigma_next * torch.randn_like(x0[:, :k_pin])
        x = x_next
    return model.normalizer.unnormalize(x)


def cmd_gen(args):
    import gym
    import d4rl  # noqa: F401
    from src.h2.common import WindowData, build_model, edit_metrics, load_cfg, pick_device
    from src.h2.rl_data import amplified_condition, windows_to_transitions
    from src.h2.rl_train import load_dataset

    name = f"accel_k{args.k}_alpha_{args.alpha:g}"
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "jax_data").mkdir(parents=True, exist_ok=True)
    if (OUT / "jax_data" / f"{name}.npz").exists():
        print(f"{name} exists")
        return
    cfg = load_cfg(CKPT)
    wd = WindowData(ENV, cfg.Dataset.seq_len, cfg.Dataset.gamma)
    device = pick_device("auto")
    model = build_model(CKPT, cfg, wd, device)
    full = dict(np.load("outputs/h2_rl/data/sources.npz"))
    ix = np.sort(np.random.default_rng(2).choice(len(full["episode"]), N_WINDOWS, replace=False))
    src = {k: v[ix] for k, v in full.items()}
    g = cfg.SimpleDiffusionGenerator
    cond = amplified_condition(src["rtg"], np.float32(args.alpha), cfg.Dataset.reward_scale)
    edited = np.empty((N_WINDOWS, wd.H + 1, wd.obs_dim + wd.act_dim + 1), np.float32)
    torch.manual_seed(0)
    for b in range(0, N_WINDOWS, args.batch):
        j = np.arange(b, min(N_WINDOWS, b + args.batch))
        x = torch.from_numpy(wd.windows(src["episode"][j], src["start"][j]))
        y = sample_pinned(model, x, torch.from_numpy(cond[j]).to(device), args.mu, args.k, g.temperature, g.num_sample_steps)
        edited[j] = y.cpu().numpy()
        print(f"{name}: {j[-1] + 1}/{N_WINDOWS}", flush=True)
    env = gym.make(ENV)
    chk = np.random.default_rng(0).choice(N_WINDOWS, 1000, replace=False)
    sources = wd.windows(src["episode"][chk], src["start"][chk])
    m = edit_metrics(env, edited[chk], sources, wd.obs_dim, wd.act_dim, wd.state_std(), desc=f"check {name}")
    v_e, v_s = edited[:, :-1, 8], np.stack([wd.eps[e]["obs"][s:s + wd.H, 8] for e, s in zip(src["episode"], src["start"])])
    report = dict(name=name, k_pin=args.k, alpha=args.alpha, mu=args.mu,
                  speed_src=float(v_s.mean()), speed_edit=float(v_e.mean()),
                  edit_first5=float(v_e[:, :5].mean()), edit_last5=float(v_e[:, -5:].mean()),
                  jump_at_start=float((v_e[:, :5] - v_s[:, :5]).mean()),
                  ramp_within=float((v_e[:, -5:] - v_e[:, :5]).mean()),
                  dyn_mse=float(m["dyn_mse"].mean()), share_broken=float((m["dyn_mse"] > 0.0085).mean()),
                  true_reward=float(m["oracle_r"].mean()))
    print(json.dumps(report, indent=1), flush=True)
    (OUT / f"check_{name}.json").write_text(json.dumps(report, indent=1))
    np.savez(OUT / "data" / f"{name}.npz", check_idx=chk, check_dyn_mse=m["dyn_mse"], check_oracle_r=m["oracle_r"],
             **windows_to_transitions(edited, wd.obs_dim, wd.act_dim))
    np.savez(OUT / "jax_data" / f"{name}.npz", **load_dataset(ENV, OUT / "data" / f"{name}.npz"))
    print(f"saved {name}", flush=True)


def _eval_one(run):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from scripts.h2_rl_jax import _evaluate
    final_steps = [960_000, 970_000, 980_000, 990_000, 1_000_000]
    rows = _evaluate(CKPT_ROOT / run, sorted(set(final_steps + list(range(100_000, 1_000_001, 100_000)))), 10, 0)
    return run, float(np.mean([v for s, v in rows if s in final_steps])), rows


def cmd_eval(args):
    from scipy.stats import ttest_ind
    runs = sorted(p.name for p in CKPT_ROOT.glob("accel__*") if (p / "actor_step_1000000.npz").exists())
    res = {}
    with ProcessPoolExecutor(min(9, max(1, len(runs)))) as ex:
        for run, final, rows in ex.map(_eval_one, runs):
            res[run] = {"final": final, "curve": rows}
            print(f"{run}: {final:.2f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    ref = json.loads(Path("outputs/h2_alpha/results.json").read_text())
    lines = ["# Acceleration edits (first steps pinned to the source), TD3+BC (JAX), score = mean of evals 960k-1M", "",
             "| condition | mean | std | seeds | unpinned GTA, same alpha |", "|---|---|---|---|---|"]
    for name in sorted({r.split("__")[1].rsplit("_seed", 1)[0] for r in res}):
        v = np.array([r["final"] for k, r in res.items() if k.startswith(f"accel__{name}_seed")])
        a = name.split("_alpha_")[1]
        u = np.array([r["final"] for k, r in ref.items() if k.startswith(f"alpha__alpha_{a}_seed") and int(k.split("seed")[1]) < 3])
        p = ttest_ind(v, u, equal_var=False).pvalue if len(v) > 1 and len(u) > 1 else float("nan")
        lines.append(f"| {name} | {v.mean():.2f} | {v.std(ddof=1) if len(v) > 1 else 0:.2f} | {len(v)} | "
                     f"{u.mean():.2f} (diff {v.mean() - u.mean():+.2f}, p = {p:.3f}) |")
    lines += ["", "References: GTA default (mu 0.5, alpha 1.3) 58.5; best tuned global 59.05; +5 target 63.5."]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("gen")
    q.add_argument("--alpha", type=float, default=1.5)
    q.add_argument("--k", type=int, default=6)
    q.add_argument("--mu", type=float, default=0.5)
    q.add_argument("--batch", type=int, default=1024)
    sub.add_parser("eval")
    args = p.parse_args()
    {"gen": cmd_gen, "eval": cmd_eval}[args.cmd](args)

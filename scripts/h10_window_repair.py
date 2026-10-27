"""Whole-window repair: project each GTA-generated window onto the learned dynamics, keeping the chain intact.

For a GTA window (s_0..s_31, a_0..a_30), optimise a start state s^_0 and actions a^_0..a^_30 (a^ = tanh(u), so in
[-1, 1]) by gradient descent through the H5 dynamics ensemble (mean of 5 members):
    s^_{t+1} = f(s^_t, a^_t)                       (chain-consistent by construction)
    loss = mean_t ||(s^_t - s_t) / sd_s||^2  +  lam * mean_t ||(a^_t - a_t) / sd_a||^2
Rewards come from the ensemble's reward head along the repaired rollout.

    python scripts/h10_window_repair.py check                 # simulator check on the 1,000 checked windows/alpha
    python scripts/h10_window_repair.py data --alphas 1.5 1.8 # repaired datasets for RL (outputs/h10_window_repair)
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pyrootutils
import torch
import torch.nn as nn

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h10_window_repair")
OBS, ACT, H, K = 17, 6, 31, 5
THETA = 0.0085
dev = "cuda"


def mlp():
    return nn.Sequential(nn.Linear(OBS + ACT, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, OBS + 1))


ck = torch.load("outputs/h5_vae_dyn/dyn.pt", map_location=dev)
ens = nn.ModuleList([mlp() for _ in range(K)]).to(dev)
ens.load_state_dict(ck["ens"])
ens.eval()
for p in ens.parameters():
    p.requires_grad_(False)
st = {k: v.to(dev) for k, v in ck["stats"].items()}
eps = pickle.load(open("data/halfcheetah-medium-v2.pkl", "rb"))
S_SD = torch.tensor(np.concatenate([e["observations"] for e in eps]).std(0) + 1e-6, device=dev)
A_SD = torch.tensor(np.concatenate([e["actions"] for e in eps]).std(0) + 1e-6, device=dev)


def f(s, a):
    x = (torch.cat([s, a], -1) - st["in_mean"]) / st["in_std"]
    y = torch.stack([m(x) for m in ens]).mean(0) * st["out_std"] + st["out_mean"]
    return s + y[:, :OBS], y[:, OBS]


def rollout(s0, a):
    s, ss, rs = s0, [s0], []
    for t in range(H):
        s, r = f(s, a[:, t])
        ss.append(s)
        rs.append(r)
    return torch.stack(ss, 1), torch.stack(rs, 1)  # [B, 32, OBS], [B, 31]


def repair(states, actions, lam, iters, lr=0.02):
    """states [B, 32, OBS] (incl. final next state), actions [B, 31, ACT] -> repaired (states, actions, rewards)."""
    tgt_s = torch.as_tensor(states, device=dev)
    tgt_a = torch.as_tensor(actions, device=dev)
    s0 = tgt_s[:, 0].clone().requires_grad_(True)
    u = torch.atanh(tgt_a.clamp(-0.999, 0.999)).clone().requires_grad_(True)
    opt = torch.optim.Adam([s0, u], lr=lr)
    for _ in range(iters):
        a = torch.tanh(u)
        ss, _ = rollout(s0, a)
        loss = (((ss - tgt_s) / S_SD) ** 2).mean() + lam * (((a - tgt_a) / A_SD) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = torch.tanh(u)
        ss, rs = rollout(s0, a)
    return ss.cpu().numpy(), a.cpu().numpy(), rs.cpu().numpy()


def f_all(s, a):
    """Ensemble mean and member variance of the next state, for [B, T, .] inputs."""
    x = (torch.cat([s, a], -1) - st["in_mean"]) / st["in_std"]
    y = torch.stack([m(x) for m in ens]) * st["out_std"] + st["out_mean"]  # [K, B, T, OBS+1]
    nxt = s[None] + y[..., :OBS]
    return nxt.mean(0), nxt.var(0), y[..., OBS].mean(0)


def repair_colloc(states, actions, lam, iters, w_dyn=100.0, w_dis=10.0, lr=0.01):
    """Collocation: every state is a free variable; one-step consistency with the ensemble as a strong penalty,
    ensemble disagreement as a penalty against model exploitation. The chain s'_t = s^_{t+1} holds exactly."""
    tgt_s = torch.as_tensor(states, device=dev)
    tgt_a = torch.as_tensor(actions, device=dev)
    S = tgt_s.clone().requires_grad_(True)
    u = torch.atanh(tgt_a.clamp(-0.999, 0.999)).clone().requires_grad_(True)
    opt = torch.optim.Adam([S, u], lr=lr)
    for _ in range(iters):
        a = torch.tanh(u)
        mean, var, _ = f_all(S[:, :H], a)
        dyn = (((S[:, 1:] - mean) / S_SD) ** 2).mean()
        loss = (((S - tgt_s) / S_SD) ** 2).mean() + lam * (((a - tgt_a) / A_SD) ** 2).mean() \
            + w_dyn * dyn + w_dis * (var / S_SD ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = torch.tanh(u)
        _, _, r = f_all(S[:, :H], a)
    return S.detach().cpu().numpy(), a.cpu().numpy(), r.cpu().numpy()


def windows_of(d, idx):
    sl = (np.asarray(idx)[:, None] * H + np.arange(H)[None]).ravel()
    s = d["observations"][sl].reshape(len(idx), H, OBS)
    last = d["next_observations"][sl].reshape(len(idx), H, OBS)[:, -1:]
    return np.concatenate([s, last], 1).astype(np.float32), d["actions"][sl].reshape(len(idx), H, ACT).astype(np.float32), \
        d["rewards"][sl].reshape(len(idx), H)


def cmd_check(args):
    import gym
    import d4rl  # noqa: F401
    env = gym.make("halfcheetah-medium-v2")
    sim = env.unwrapped
    nq = sim.model.nq
    sstd = S_SD.cpu().numpy()

    def true_err(states, actions):
        e = np.empty(states.shape[:2][0])
        speeds, rews = [], []
        for i in range(len(states)):
            errs = []
            for t in range(H):
                s = states[i, t].astype(np.float64)
                sim.set_state(np.concatenate([[0.0], s[:nq - 1]]), s[nq - 1:])
                o, r, _, _ = sim.step(actions[i, t])
                errs.append((((o - states[i, t + 1]) / sstd) ** 2).mean())
                rews.append(r)
            e[i] = np.mean(errs)
        return e, float(np.mean(rews))

    rows = []
    for al in args.alphas:
        d = np.load(f"outputs/h2_alpha/data/alpha_{al}.npz")
        idx = d["check_idx"][: args.n]
        s, a, r = windows_of(d, idx)
        e_gta, tr_gta = true_err(s, a)
        for lam in args.lams:
            rs, ra, rr = (repair_colloc if args.mode == 'colloc' else repair)(s, a, lam, args.iters)
            e_rep, tr_rep = true_err(rs, ra)
            row = dict(alpha=al, lam=lam, mode=args.mode,
                       speed_gta=float(s[:, :H, 8].mean()), speed_rep=float(rs[:, :H, 8].mean()),
                       true_reward_gta=tr_gta, true_reward_rep=tr_rep, model_reward_rep=float(rr.mean()),
                       dyn_gta=float(e_gta.mean()), over_gta=float((e_gta > THETA).mean()),
                       dyn_rep=float(e_rep.mean()), over_rep=float((e_rep > THETA).mean()),
                       track_state=float(np.sqrt((((rs - s) / sstd) ** 2).mean())),
                       action_change=float(np.abs(ra - a).mean()))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / f"check_{args.mode}.json", "w"), indent=1)


def cmd_data(args):
    from src.h2.rl_train import load_dataset
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "jax_data").mkdir(parents=True, exist_ok=True)
    for al in args.alphas:
        d = dict(np.load(f"outputs/h2_alpha/data/alpha_{al}.npz"))
        n = len(d["observations"]) // H
        S, A, R = [], [], []
        for b in range(0, n, args.batch):
            idx = np.arange(b, min(n, b + args.batch))
            s, a, _ = windows_of(d, idx)
            rs, ra, rr = (repair_colloc if args.mode == 'colloc' else repair)(s, a, args.lams[0], args.iters)
            S.append(rs), A.append(ra), R.append(rr)
            print(f"alpha {al}: {b + len(idx)}/{n} windows", flush=True)
        S, A, R = np.concatenate(S), np.concatenate(A), np.concatenate(R)
        d["observations"] = S[:, :H].reshape(-1, OBS)
        d["next_observations"] = S[:, 1:].reshape(-1, OBS)
        d["actions"] = A.reshape(-1, ACT)
        d["rewards"] = R.reshape(-1)
        tag = f"wrepair_alpha_{al}"
        np.savez(OUT / "data" / f"{tag}.npz", **d)
        np.savez(OUT / "jax_data" / f"{tag}.npz", **load_dataset("halfcheetah-medium-v2", OUT / "data" / f"{tag}.npz"))
        print(f"saved {tag}: speed {d['observations'][:, 8].mean():.2f}, reward {d['rewards'].mean():.2f}", flush=True)


def _eval_one(job):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from scripts.h2_rl_jax import _evaluate
    run, upto = job
    final_steps = [upto - 40_000 + 10_000 * k for k in range(5)]
    rows = _evaluate(Path("jax_impl/checkpoints") / run, sorted(set(final_steps + list(range(100_000, upto + 1, 100_000)))), 10, 0)
    return run, float(np.mean([v for s, v in rows if s in final_steps])), rows


def cmd_eval(args):
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import ttest_ind
    ck = Path("jax_impl/checkpoints")
    runs = sorted(p.name for p in ck.glob("wrepair__*") if (p / f"actor_step_{args.upto:07d}.npz").exists())
    res = {}
    with ProcessPoolExecutor(min(9, max(1, len(runs)))) as ex:
        for run, final, rows in ex.map(_eval_one, [(r, args.upto) for r in runs]):
            res[run] = {"final": final, "curve": rows}
            print(f"{run}: {final:.2f}", flush=True)
    (OUT / f"results_{args.upto}.json").write_text(json.dumps(res, indent=1))
    ref = json.loads(Path("outputs/h2_alpha/results.json").read_text())
    lines = [f"# Whole-window (collocation) repaired GTA data, TD3+BC (JAX); score = mean of evals at "
             f"{args.upto - 40000}..{args.upto} steps", "",
             "| condition | mean | std | seeds | unrepaired GTA, same alpha (1M steps) |", "|---|---|---|---|---|"]
    for al in args.alphas:
        v = np.array([r["final"] for k, r in res.items() if k.startswith(f"wrepair__wrepair_alpha_{al}_seed")])
        u = np.array([r["final"] for k, r in ref.items() if k.startswith(f"alpha__alpha_{al}_seed") and int(k.split("seed")[1]) < 3])
        if len(v):
            p = ttest_ind(v, u, equal_var=False).pvalue if len(v) > 1 and len(u) > 1 else float("nan")
            lines.append(f"| repaired alpha {al} | {v.mean():.2f} | {v.std(ddof=1) if len(v) > 1 else 0:.2f} | {len(v)} | "
                         f"{u.mean():.2f} (diff {v.mean() - u.mean():+.2f}, p = {p:.3f}) |")
    lines += ["", "References: GTA default (mu 0.5, alpha 1.3) 58.51-58.66; best tuned global 59.05; "
              "oracle-weighted alpha 1.5 59.56; +5 target 63.5."]
    (OUT / f"summary_{args.upto}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--upto", type=int, default=1_000_000)
    e.add_argument("--alphas", nargs="+", default=["1.5", "1.8", "2"])
    for name in ["check", "data"]:
        q = sub.add_parser(name)
        q.add_argument("--alphas", nargs="+", default=["1.5", "1.8"])
        q.add_argument("--lams", nargs="+", type=float, default=[0.1])
        q.add_argument("--iters", type=int, default=300)
        q.add_argument("--n", type=int, default=300)
        q.add_argument("--batch", type=int, default=4096)
        q.add_argument("--mode", choices=["shoot", "colloc"], default="colloc")
    args = p.parse_args()
    {"check": cmd_check, "data": cmd_data, "eval": cmd_eval}[args.cmd](args)

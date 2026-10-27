"""Repair check: keep GTA's generated (s, a), replace the generated next state and reward by the dynamics
ensemble's prediction. Does that make fast (high-alpha) transitions physically correct while keeping them fast?

For the 1,000 simulator-checked windows of each alpha dataset (mu 0.5), every transition (s_t, a_t) is replayed in
MuJoCo to get the true s'_t and r_t, and compared with
  GTA       the generated s'_t, r_t
  repaired  the ensemble's s'_t, r_t  (mean of 5 members)
Baseline: the ensemble's own error on 31,000 random ORIGINAL dataset transitions.

    python scripts/h9_repair_check.py      # gta env
"""
import json
from pathlib import Path

import gym
import d4rl  # noqa: F401
import numpy as np
import pandas as pd
import pyrootutils
import torch
import torch.nn as nn

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from src.h2.common import WindowData  # noqa: E402

OUT = Path("outputs/h9_learned_trust")
OBS, ACT, H, K = 17, 6, 31, 5
THETA = 0.0085
dev = "cuda" if torch.cuda.is_available() else "cpu"


def mlp():
    return nn.Sequential(nn.Linear(OBS + ACT, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, OBS + 1))


ck = torch.load("outputs/h5_vae_dyn/dyn.pt", map_location=dev)
ens = nn.ModuleList([mlp() for _ in range(K)]).to(dev)
ens.load_state_dict(ck["ens"])
ens.eval()
st = {k: v.to(dev) for k, v in ck["stats"].items()}


@torch.no_grad()
def model_step(s, a):
    s_t, a_t = torch.as_tensor(s, device=dev), torch.as_tensor(a, device=dev)
    x = (torch.cat([s_t, a_t], -1) - st["in_mean"]) / st["in_std"]
    y = torch.stack([m(x) for m in ens]).mean(0) * st["out_std"] + st["out_mean"]
    return (s_t + y[:, :OBS]).cpu().numpy(), y[:, OBS].cpu().numpy()


env = gym.make("halfcheetah-medium-v2")
sim = env.unwrapped
nq = sim.model.nq
wd = WindowData("halfcheetah-medium-v2", H, 0.99)
sstd = wd.state_std()


def true_step(s, a):
    nxt, rew = np.empty_like(s), np.empty(len(s), np.float32)
    for i in range(len(s)):
        sim.set_state(np.concatenate([[0.0], s[i, :nq - 1]]), s[i, nq - 1:])
        o, r, _, _ = sim.step(a[i])
        nxt[i], rew[i] = o, r
    return nxt, rew


def summarise(name, s, a, s_gen, r_gen):
    s_true, r_true = true_step(s, a)
    s_rep, r_rep = model_step(s, a)
    err = lambda x: (((x - s_true) / sstd) ** 2).mean(1)  # noqa: E731  per-transition dyn MSE
    e_gen, e_rep = err(s_gen), err(s_rep)
    win = lambda e: e.reshape(-1, H).mean(1)  # noqa: E731  per-window mean, as dyn_mse elsewhere
    out = dict(set=name, speed=float(s[:, 8].mean()), true_reward_per_step=float(r_true.mean()),
               gta_dyn=float(win(e_gen).mean()), gta_share_over=float((win(e_gen) > THETA).mean()),
               rep_dyn=float(win(e_rep).mean()), rep_share_over=float((win(e_rep) > THETA).mean()),
               gta_reward_err=float(np.abs(r_gen - r_true).mean()), rep_reward_err=float(np.abs(r_rep - r_true).mean()))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out


rows = []
# baseline: the ensemble on original data (s, a, s', r all real)
rng = np.random.default_rng(0)
e_ix = rng.integers(0, len(wd.eps), 1000)
s_ix = rng.integers(0, 1000 - H, 1000)
W = wd.windows(e_ix, s_ix)
s, a = W[:, :H, :OBS].reshape(-1, OBS), W[:, :H, OBS:OBS + ACT].reshape(-1, ACT)
s_next = np.concatenate([W[:, 1:H, :OBS], W[:, H:, :OBS]], 1).reshape(-1, OBS)
rows.append(summarise("original data", s.astype(np.float32), a.astype(np.float32), s_next,
                      W[:, :H, OBS + ACT].reshape(-1)))

for al in ["1.3", "1.5", "1.8", "2"]:
    d = np.load(f"outputs/h2_alpha/data/alpha_{al}.npz")
    sl = (d["check_idx"][:, None] * H + np.arange(H)[None]).ravel()
    rows.append(summarise(f"alpha {al}", d["observations"][sl], d["actions"][sl], d["next_observations"][sl],
                          d["rewards"][sl]))

df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "repair_check.csv", index=False)
json.dump(rows, open(OUT / "repair_check.json", "w"), indent=1)
print(df.round(4).to_string(index=False))

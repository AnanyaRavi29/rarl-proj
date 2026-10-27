"""H4: latent trajectory editing (VAE) as a cheap replacement for GTA's diffusion model.

A VAE encodes a 31-step window (+ final next-state, 32 x 24) into z (64-d); a head predicts the window's quality
q (discounted reward over its 31 steps) from z. Editing = find the latent closest to the window's own code whose
predicted quality is alpha * q, then decode:
    z* = argmin_z (head(z) - target)^2 + lam * ||z - z0||^2 / dim,   target = alpha * q (standardised)

    python scripts/h4_vae.py train     # ~15-20 min on GPU
    python scripts/h4_vae.py check     # simulator check of edits on 1,000 windows, several alpha / lam
    python scripts/h4_vae.py gen --alpha 1.3 --lam 0.1     # 1M transitions -> JAX data file
    python scripts/h4_vae.py eval / report                 # after JAX TD3+BC (see h4_vae.sh)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h4_vae")
ENV = "halfcheetah-medium-v2"
H, OBS, ACT = 31, 17, 6
D = OBS + ACT + 1
ZDIM = 64


def load_arrays():
    """All transitions as flat arrays + per-window quality for every valid start (episodes are 1000 steps)."""
    import pickle
    with open(f"data/{ENV}.pkl", "rb") as f:
        eps = pickle.load(f)
    obs = np.concatenate([e["observations"] for e in eps]).astype(np.float32)
    act = np.concatenate([e["actions"] for e in eps]).astype(np.float32)
    rew = np.concatenate([np.asarray(e["rewards"]).reshape(-1) for e in eps]).astype(np.float32)
    nxt = np.concatenate([e["next_observations"] for e in eps]).astype(np.float32)
    lens = np.array([len(e["rewards"]) for e in eps])
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    starts = np.concatenate([o + np.arange(l - H + 1) for o, l in zip(offs, lens)])  # global index of window start
    kern = 0.99 ** np.arange(H)
    q_all = np.concatenate([np.convolve(np.asarray(e["rewards"], np.float64).reshape(-1), kern[::-1], "valid")
                            for e in eps]).astype(np.float32)
    return obs, act, rew, nxt, starts, q_all, offs


class Data:
    def __init__(self, device):
        import torch
        obs, act, rew, nxt, starts, q, offs = load_arrays()
        self.offs = offs
        self.rows = torch.from_numpy(np.concatenate([obs, act, rew[:, None]], 1)).to(device)
        self.nxt = torch.from_numpy(nxt).to(device)
        self.mean = self.rows.mean(0)
        self.std = self.rows.std(0) + 1e-6
        self.starts = torch.from_numpy(starts).to(device)
        self.q = torch.from_numpy(q).to(device)
        self.q_mean, self.q_std = self.q.mean(), self.q.std()
        self.ar = torch.arange(H, device=device)
        mask = torch.ones(H + 1, D, device=device)
        mask[-1, OBS:] = 0
        self.mask = mask

    def windows(self, idx):
        """idx: positions in self.starts -> raw windows [B, 32, 24]."""
        import torch
        g = self.starts[idx][:, None] + self.ar[None]
        w = self.rows[g]
        last = torch.zeros(len(idx), 1, D, device=w.device)
        last[:, 0, :OBS] = self.nxt[g[:, -1]]
        return torch.cat([w, last], 1)

    def norm(self, w):
        return (w - self.mean) / self.std * self.mask

    def unnorm(self, x):
        w = x * self.std + self.mean
        w[:, -1, OBS:] = 0
        return w


def build_vae(device):
    import torch.nn as nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            n = (H + 1) * D
            self.enc = nn.Sequential(nn.Linear(n, 1024), nn.SiLU(), nn.Linear(1024, 512), nn.SiLU(), nn.Linear(512, 2 * ZDIM))
            self.dec = nn.Sequential(nn.Linear(ZDIM, 512), nn.SiLU(), nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, n))
            self.head = nn.Sequential(nn.Linear(ZDIM, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

        def encode(self, x):
            mu, logvar = self.enc(x.flatten(1)).chunk(2, -1)
            return mu, logvar.clamp(-10, 5)

        def decode(self, z):
            return self.dec(z).view(-1, H + 1, D)

    return VAE().to(device)


def cmd_train(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    data = Data(dev)
    vae = build_vae(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(data.starts), (1024,), device=dev)
        x = data.norm(data.windows(idx))
        mu, logvar = vae.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = (((vae.decode(z) - x) ** 2) * data.mask).sum((1, 2)).mean() / data.mask.sum()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1).mean() / ZDIM
        qt = (data.q[idx] - data.q_mean) / data.q_std
        qh = ((vae.head(mu).squeeze(1) - qt) ** 2).mean()
        loss = rec + args.beta * kl + qh
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == 1:
            print(f"step {step:6d} | recon {rec.item():.4f} | kl {kl.item():.3f} | quality-head mse {qh.item():.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save(vae.state_dict(), OUT / "vae.pt")
    print(f"saved {OUT / 'vae.pt'}")


def edit(vae, data, idx, alpha, lam, iters=200, lr=0.05):
    """Nearest latent to the window's code with predicted quality alpha * q; returns raw edited windows."""
    import torch
    with torch.no_grad():
        x = data.norm(data.windows(idx))
        z0, _ = vae.encode(x)
    if alpha == 1.0 and lam is None:  # pure reconstruction
        with torch.no_grad():
            return data.unnorm(vae.decode(z0))
    target = (alpha * data.q[idx] - data.q_mean) / data.q_std
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        loss = ((vae.head(z).squeeze(1) - target) ** 2).mean() + lam * ((z - z0) ** 2).sum(1).mean() / ZDIM
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return data.unnorm(vae.decode(z))


def _setup():
    import torch
    import gym
    import d4rl  # noqa: F401
    data = Data("cuda")
    vae = build_vae("cuda")
    vae.load_state_dict(torch.load(OUT / "vae.pt"))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return data, vae, gym.make(ENV)


def cmd_check(args):
    import torch
    from src.h2.common import edit_metrics
    data, vae, env = _setup()
    std = data.std[:OBS].cpu().numpy()
    torch.manual_seed(0)
    idx = torch.randint(0, len(data.starts), (1000,), device="cuda")
    src = data.windows(idx).cpu().numpy()
    m0 = edit_metrics(env, src, src, OBS, ACT, std, desc="sources")
    rows = [("source windows", "-", "-", m0)]
    settings = [(1.0, None)] + [(a, l) for a in [1.1, 1.3, 1.5] for l in [0.01, 0.1, 1.0]]
    for a, l in settings:
        w = edit(vae, data, idx, a, l).cpu().numpy()
        rows.append(("reconstruction" if l is None else "edit", a, l, edit_metrics(env, w, src, OBS, ACT, std, desc=f"a{a} l{l}")))
    lines = ["# H4 VAE editing: simulator check on 1,000 windows", "",
             "GTA reference (alpha 1.3, mu 0.5): edited reward +32% vs source, dynamic MSE 0.0075, shift 0.89.", "",
             "| method | alpha | lam | edited reward | gain vs source | model reward - true | dynamic MSE | shift |",
             "|---|---|---|---|---|---|---|---|"]
    base = m0["oracle_r"].mean()
    for name, a, l, m in rows:
        lines.append(f"| {name} | {a} | {l} | {m['oracle_r'].mean():.1f} | {100 * (m['oracle_r'].mean() / base - 1):+.1f}% | "
                     f"{(m['gen_r'] - m['oracle_r']).mean():+.2f} | {m['dyn_mse'].mean():.4f} | {m['shift'].mean():.2f} |")
    (OUT / "check.md").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_gen(args):
    import torch
    from src.h2.rl_data import windows_to_transitions
    from src.h2.rl_train import load_dataset
    data, vae, env = _setup()
    src = dict(np.load("outputs/h2_rl/data/sources.npz"))  # same windows as the GTA alpha runs (seed 2, 32,258)
    ix = np.sort(np.random.default_rng(2).choice(len(src["episode"]), 32_258, replace=False))
    glob_start = data.offs[src["episode"][ix]] + src["start"][ix]
    pos = torch.from_numpy(np.searchsorted(data.starts.cpu().numpy(), glob_start)).cuda()
    t0 = time.time()
    out = []
    for b in range(0, len(pos), 4096):
        out.append(edit(vae, data, pos[b:b + 4096], args.alpha, args.lam).cpu().numpy())
    edited = np.concatenate(out)
    print(f"generated {len(edited)} windows in {time.time() - t0:.1f}s")
    name = f"vae_a{args.alpha:g}_l{args.lam:g}"
    gen = OUT / "data" / f"{name}_gen.npz"
    gen.parent.mkdir(parents=True, exist_ok=True)
    np.savez(gen, **windows_to_transitions(edited, OBS, ACT))
    (OUT / "jax_data").mkdir(exist_ok=True)
    np.savez(OUT / "jax_data" / f"{name}.npz", **load_dataset(ENV, gen))
    print(f"{name}: JAX data ready")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "check", "gen"])
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=1.3)
    p.add_argument("--lam", type=float, default=0.1)
    a = p.parse_args()
    {"train": cmd_train, "check": cmd_check, "gen": cmd_gen}[a.cmd](a)

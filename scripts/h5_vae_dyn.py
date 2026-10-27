"""H5: VAE-edited actions + learned dynamics for the physics (TGCVG-style fix of H4).

Generation for a source window: keep its real first state s_0, take the VAE-edited actions a'_0..a'_30
(scripts/h4_vae.py: latent edit towards alpha x quality), and roll them through a learned dynamics ensemble:
    s_{t+1} = s_t + mean_k f_k(s_t, a'_t)[:17],   r_t = mean_k f_k(s_t, a'_t)[17]
so states and rewards come from a model trained for dynamics, not from the VAE decoder.

    python scripts/h5_vae_dyn.py train     # dynamics ensemble (~3 min)
    python scripts/h5_vae_dyn.py check     # simulator check on 1,000 windows
    python scripts/h5_vae_dyn.py gen --alpha 1.3 --lam 1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from scripts.h4_vae import ACT, ENV, H, OBS, Data, _setup, edit  # noqa: E402

OUT = Path("outputs/h5_vae_dyn")
K = 5


def build_ens(device):
    import torch.nn as nn

    def mlp():
        return nn.Sequential(nn.Linear(OBS + ACT, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                             nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, OBS + 1))
    return nn.ModuleList([mlp() for _ in range(K)]).to(device)


class Dyn:
    """Normalised (s, a) -> normalised (delta s, r); ensemble mean prediction."""
    def __init__(self, data, ens, stats):
        self.data, self.ens, self.st = data, ens, stats

    def step(self, s, a):
        import torch
        x = (torch.cat([s, a], -1) - self.st["in_mean"]) / self.st["in_std"]
        y = torch.stack([m(x) for m in self.ens]).mean(0) * self.st["out_std"] + self.st["out_mean"]
        return s + y[:, :OBS], y[:, OBS]


def cmd_train(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data("cuda")
    rows, nxt = data.rows, data.nxt
    x = rows[:, :OBS + ACT]
    y = torch.cat([nxt - rows[:, :OBS], rows[:, OBS + ACT:]], 1)
    st = dict(in_mean=x.mean(0), in_std=x.std(0) + 1e-6, out_mean=y.mean(0), out_std=y.std(0) + 1e-6)
    xn, yn = (x - st["in_mean"]) / st["in_std"], (y - st["out_mean"]) / st["out_std"]
    perm = torch.randperm(len(xn), device="cuda")
    val, tr = perm[:20_000], perm[20_000:]
    ens = build_ens("cuda")
    opt = torch.optim.Adam(ens.parameters(), lr=1e-3, weight_decay=1e-5)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        loss = 0
        for m in ens:  # each member sees its own bootstrap batch
            b = tr[torch.randint(0, len(tr), (1024,), device="cuda")]
            loss = loss + ((m(xn[b]) - yn[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5000 == 0:
            with torch.no_grad():
                vl = torch.stack([((m(xn[val]) - yn[val]) ** 2).mean() for m in ens]).mean()
            print(f"step {step} | train {loss.item() / K:.4f} | val {vl.item():.4f} | {time.time() - t0:.0f}s", flush=True)
    torch.save({"ens": ens.state_dict(), "stats": {k: v.cpu() for k, v in st.items()}}, OUT / "dyn.pt")
    print(f"saved {OUT / 'dyn.pt'}")


def _load_dyn(data):
    import torch
    ck = torch.load(OUT / "dyn.pt")
    ens = build_ens("cuda")
    ens.load_state_dict(ck["ens"])
    ens.eval()
    for p in ens.parameters():
        p.requires_grad_(False)
    return Dyn(data, ens, {k: v.cuda() for k, v in ck["stats"].items()})


def rollout(dyn, data, idx, actions):
    """Real first state + given actions [B, 31, 6] -> window [B, 32, 24] built by the dynamics model."""
    import torch
    with torch.no_grad():
        s = data.windows(idx)[:, 0, :OBS]
        states, rews = [s], []
        for t in range(H):
            s, r = dyn.step(s, actions[:, t])
            states.append(s)
            rews.append(r)
        S = torch.stack(states, 1)
        A = torch.cat([actions, torch.zeros_like(actions[:, :1])], 1)
        R = torch.cat([torch.stack(rews, 1), torch.zeros_like(rews[0][:, None])], 1)[..., None]
        return torch.cat([S, A, R], -1)


def make(dyn, data, vae, idx, alpha, lam):
    import torch
    if alpha is None:  # control: original actions through the dynamics model
        acts = data.windows(idx)[:, :H, OBS:OBS + ACT]
    else:
        acts = edit(vae, data, idx, alpha, lam)[:, :H, OBS:OBS + ACT].clamp(-1, 1)
    return rollout(dyn, data, idx, acts)


def cmd_check(args):
    import torch
    from src.h2.common import edit_metrics
    data, vae, env = _setup()
    dyn = _load_dyn(data)
    std = data.std[:OBS].cpu().numpy()
    torch.manual_seed(0)
    idx = torch.randint(0, len(data.starts), (1000,), device="cuda")
    src = data.windows(idx).cpu().numpy()
    base = edit_metrics(env, src, src, OBS, ACT, std, desc="src")["oracle_r"].mean()
    lines = ["# H5: VAE-edited actions + dynamics-ensemble rollouts (1,000 windows)", "",
             f"Source windows: true reward {base:.1f}. GTA reference: +32%, dynamic MSE 0.0075. H4 (VAE states): +28%, 0.046.", "",
             "| actions | alpha | lam | true reward | gain | model reward - true | dynamic MSE | shift |", "|---|---|---|---|---|---|---|---|"]
    for a, l in [(None, None)] + [(a, l) for a in [1.1, 1.3, 1.5] for l in [0.1, 1.0]]:
        w = make(dyn, data, vae, idx, a, l).cpu().numpy()
        m = edit_metrics(env, w, src, OBS, ACT, std, desc=f"a{a}")
        lines.append(f"| {'original' if a is None else 'VAE-edited'} | {a or '-'} | {l or '-'} | {m['oracle_r'].mean():.1f} | "
                     f"{100 * (m['oracle_r'].mean() / base - 1):+.1f}% | {(m['gen_r'] - m['oracle_r']).mean():+.2f} | "
                     f"{m['dyn_mse'].mean():.4f} | {m['shift'].mean():.2f} |")
    (OUT / "check.md").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_gen(args):
    import torch
    from src.h2.rl_data import windows_to_transitions
    from src.h2.rl_train import load_dataset
    data, vae, env = _setup()
    dyn = _load_dyn(data)
    src = dict(np.load("outputs/h2_rl/data/sources.npz"))
    ix = np.sort(np.random.default_rng(2).choice(len(src["episode"]), 32_258, replace=False))
    pos = torch.from_numpy(np.searchsorted(data.starts.cpu().numpy(), data.offs[src["episode"][ix]] + src["start"][ix])).cuda()
    t0 = time.time()
    w = np.concatenate([make(dyn, data, vae, pos[b:b + 4096], args.alpha, args.lam).cpu().numpy()
                        for b in range(0, len(pos), 4096)])
    print(f"generated {len(w)} windows in {time.time() - t0:.1f}s")
    name = f"vaedyn_a{args.alpha:g}_l{args.lam:g}"
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "jax_data").mkdir(exist_ok=True)
    gen = OUT / "data" / f"{name}_gen.npz"
    np.savez(gen, **windows_to_transitions(w, OBS, ACT))
    np.savez(OUT / "jax_data" / f"{name}.npz", **load_dataset(ENV, gen))
    print(f"{name}: JAX data ready")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "check", "gen"])
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--alpha", type=float, default=1.3)
    p.add_argument("--lam", type=float, default=1.0)
    a = p.parse_args()
    {"train": cmd_train, "check": cmd_check, "gen": cmd_gen}[a.cmd](a)

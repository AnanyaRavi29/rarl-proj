"""Small, readable walk-through of GTA Stage B (paper Sec. 4.2) on a few windows.

Uses the trained diffusion model to edit a handful of H=31 windows from
halfcheetah-medium-v2 at several noise ratios mu, then checks every edited
transition against the real MuJoCo simulator and saves plots / a GIF.

    python scripts/gta_edit_demo.py --ckpt results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt
"""
import argparse
import pickle
from pathlib import Path

import gym
import d4rl  # noqa: F401  (registers the D4RL envs)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True)

from src.data.norm import normalizer_factory
from src.diffusion.utils import construct_diffusion_model

ENV = "halfcheetah-medium-v2"
OBS, ACT = 17, 6


def discounted_cumsum(x, gamma):
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in range(len(x) - 2, -1, -1):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def load_model(ckpt, cfg, episodes, device):
    # Same normalizer the training run built: mean/std over every (s, a, r) in the dataset.
    obs = np.concatenate([e["observations"] for e in episodes])
    act = np.concatenate([e["actions"] for e in episodes])
    rew = np.concatenate([e["rewards"] for e in episodes]).reshape(-1, 1)
    inputs = torch.from_numpy(np.concatenate([obs, act, rew], -1)).float()
    normalizer = normalizer_factory(cfg.Dataset.normalizer_type, inputs, skip_dims=[])

    T = cfg.Dataset.seq_len + 1  # 31 transitions + final next-state = 32 rows
    data_shape = {"observations": (T, OBS), "actions": (T, ACT), "rewards": (T, 1)}
    model = construct_diffusion_model(
        data_shape=data_shape,
        normalizer=normalizer,
        denoising_network=cfg.construct_diffusion_model.denoising_network,
        edm_config=cfg.ElucidatedDiffusion,
        cond_dim=cfg.construct_diffusion_model.denoising_network.cond_dim,
    )
    # GTA samples with the EMA (smoothed) weights, stored under "ema_model." in the checkpoint.
    ema = torch.load(ckpt, map_location="cpu")["ema"]
    state = {k[len("ema_model."):]: v for k, v in ema.items() if k.startswith("ema_model.")}
    model.load_state_dict(state)
    return model.to(device).eval()


def make_window(ep, start, H, gamma):
    """One source window tau^0 as a [32, 24] array plus its GTA conditioning return."""
    s = ep["observations"][start:start + H]
    a = ep["actions"][start:start + H]
    r = ep["rewards"][start:start + H].reshape(-1, 1)
    last = np.concatenate([ep["next_observations"][start + H - 1], np.zeros(ACT + 1)])
    window = np.concatenate([np.concatenate([s, a, r], -1), last[None]], 0)
    rtg = discounted_cumsum(ep["rewards"].astype(np.float64), gamma)[start]
    return window.astype(np.float32), float(rtg)


def oracle_check(env, window):
    """Replay each generated (s, a) in MuJoCo. Returns the true next states and true rewards."""
    sim = env.unwrapped
    true_next, true_r = [], []
    for t in range(window.shape[0] - 1):
        s, a = window[t, :OBS], window[t, OBS:OBS + ACT]
        qpos = np.concatenate([[0.0], s[:8]])  # x-position is not observed; it doesn't affect dynamics
        sim.set_state(qpos, s[8:])
        nxt, r, _, _ = sim.step(a)
        true_next.append(nxt)
        true_r.append(r)
    return np.array(true_next), np.array(true_r)


def render_frames(env, states, size=240):
    """Kinematic playback: put the cheetah in each state and take a picture."""
    sim = env.unwrapped
    frames = []
    for s in states:
        sim.set_state(np.concatenate([[0.0], s[:8]]), s[8:])
        frames.append(sim.render(mode="rgb_array", width=size, height=size))
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/gta.yaml")
    p.add_argument("--mus", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n_windows", type=int, default=4)
    p.add_argument("--out", default="outputs/gta_demo")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    H, gamma = cfg.Dataset.seq_len, cfg.Dataset.gamma
    alpha = cfg.SimpleDiffusionGenerator.amplify_returnscale
    reward_scale = cfg.Dataset.reward_scale
    temperature = cfg.SimpleDiffusionGenerator.temperature

    with open(f"data/{ENV}.pkl", "rb") as f:
        episodes = pickle.load(f)
    env = gym.make(ENV)

    print(f"[1] Loading trained diffusion model on {device} ...")
    model = load_model(args.ckpt, cfg, episodes, device)

    # [2] Pick source windows tau^0: random episode, random start (not near the episode end).
    picks = [(int(rng.integers(len(episodes))), int(rng.integers(0, 1000 - H - 100)))
             for _ in range(args.n_windows)]
    windows, rtgs = zip(*(make_window(episodes[e], s, H, gamma) for e, s in picks))
    tau0 = torch.from_numpy(np.stack(windows)).to(device)  # [B, 32, 24]
    # Amplified return guidance (paper Eq. 8): condition on alpha * original return.
    cond = torch.tensor(rtgs, dtype=torch.float32, device=device).reshape(-1, 1) * alpha * reward_scale
    print(f"[2] {len(picks)} source windows, shape {tuple(tau0.shape)}; "
          f"condition = {alpha} x return-to-go x {reward_scale}")

    # [3] Partial noising + guided denoising, once per mu (paper Eq. 9-10).
    edited = {}
    for mu in args.mus:
        print(f"[3] mu={mu}: jump to sigma at step {int(128 * (1 - mu))}/128, "
              f"then denoise {128 - int(128 * (1 - mu))} steps")
        with torch.no_grad():
            x = model.sample_back_and_forth(samples=tau0.clone(), cond=cond, noise_level=mu,
                                            temperature=temperature, disable_tqdm=True)
        edited[mu] = x.cpu().numpy()

    # [4] Check edited windows against the real simulator (paper App. D metrics).
    src = tau0.cpu().numpy()
    s_std = np.concatenate([e["observations"] for e in episodes]).std(0) + 1e-6
    rows = []
    for i, (e, st) in enumerate(picks):
        _, r_orig = oracle_check(env, src[i])
        rows.append(dict(window=i, episode=e, start=st, mu=0.0, gen_r=src[i, :-1, OBS + ACT].sum(),
                         oracle_r=r_orig.sum(), dyn_mse=0.0, shift=0.0))
        for mu, x in edited.items():
            true_next, true_r = oracle_check(env, x[i])
            gen_next = x[i, 1:, :OBS]
            dyn = (((true_next - gen_next) / s_std) ** 2).mean()
            shift = np.sqrt((((x[i, :-1, :OBS] - src[i, :-1, :OBS]) / s_std) ** 2).mean())
            rows.append(dict(window=i, episode=e, start=st, mu=mu, gen_r=x[i, :-1, OBS + ACT].sum(),
                             oracle_r=true_r.sum(), dyn_mse=dyn, shift=shift))

    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print("\n[4] Per window (sums over 31 steps). mu=0 is the original data.")
    print(f"{'win':>3} {'mu':>5} {'gen reward':>11} {'oracle reward':>14} {'dyn MSE':>9} {'state shift':>12}")
    for r in rows:
        print(f"{r['window']:>3} {r['mu']:>5.2f} {r['gen_r']:>11.2f} {r['oracle_r']:>14.2f} "
              f"{r['dyn_mse']:>9.4f} {r['shift']:>12.3f}")

    # [5] Plots: x-velocity (obs[8]) and reward over the 31 steps, original vs each mu.
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(args.mus)))
    fig, axes = plt.subplots(2, args.n_windows, figsize=(4 * args.n_windows, 6), squeeze=False)
    t = np.arange(H)
    for i in range(args.n_windows):
        axes[0, i].plot(t, src[i, :-1, 8], "k", lw=2, label="original")
        axes[1, i].plot(t, src[i, :-1, OBS + ACT], "k", lw=2, label="original")
        for c, (mu, x) in zip(colors, edited.items()):
            axes[0, i].plot(t, x[i, :-1, 8], color=c, label=f"mu={mu}")
            axes[1, i].plot(t, x[i, :-1, OBS + ACT], color=c, label=f"mu={mu}")
        axes[0, i].set_title(f"window {i} (ep {picks[i][0]}, t={picks[i][1]})")
    axes[0, 0].set_ylabel("forward velocity (obs[8])")
    axes[1, 0].set_ylabel("reward")
    for ax in axes[1]:
        ax.set_xlabel("step in window")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "windows_mu_sweep.png", dpi=120)
    print(f"\n[5] Saved {out / 'windows_mu_sweep.png'}")

    # [6] GIF: original (left) vs mu=0.5 edit (right) for window 0.
    try:
        import imageio
        mu_vid = 0.5 if 0.5 in edited else args.mus[0]
        left = render_frames(env, src[0, :, :OBS])
        right = render_frames(env, edited[mu_vid][0, :, :OBS])
        frames = [np.concatenate([l, r], 1) for l, r in zip(left, right)]
        imageio.mimsave(out / "window0_original_vs_edit.gif", frames, duration=0.08)
        print(f"[6] Saved {out / 'window0_original_vs_edit.gif'} (left original, right mu={mu_vid})")
    except Exception as exc:  # rendering needs an offscreen OpenGL context
        print(f"[6] Skipped GIF: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

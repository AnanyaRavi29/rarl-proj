"""Shared helpers: dataset windows, model loading, device choice, simulator checks."""
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.signal import lfilter

from src.data.norm import normalizer_factory
from src.diffusion.utils import construct_diffusion_model


def discounted_cumsum(x, gamma):
    return lfilter([1.0], [1.0, -gamma], x[::-1])[::-1].copy()


def pick_device(device="auto", min_free_gb=3.0):
    """Use the GPU only if it has enough free memory (the GPU on this machine is shared)."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free / 1e9 >= min_free_gb:
                return "cuda"
            print(f"GPU has only {free / 1e9:.1f} GB free -> using CPU")
        except RuntimeError as exc:
            print(f"GPU unusable ({exc}) -> using CPU")
    return "cpu"


def load_cfg(ckpt, config=None):
    path = Path(config) if config else Path(ckpt).parent / "config.yaml"
    if not path.exists():
        path = Path("configs/gta.yaml")
    print(f"Config: {path}")
    return OmegaConf.load(path)


class WindowData:
    """Episodes from data/<env>.pkl, cut into GTA source windows of H transitions (+ final next-state)."""

    def __init__(self, env_name, H, gamma, penalty=100.0):
        with open(f"data/{env_name}.pkl", "rb") as f:
            episodes = pickle.load(f)
        self.H = H
        self.obs_dim = episodes[0]["observations"].shape[1]
        self.act_dim = episodes[0]["actions"].shape[1]
        self.eps = []
        for ep in episodes:
            rew = np.asarray(ep["rewards"], dtype=np.float64).reshape(-1).copy()
            if np.asarray(ep["terminals"]).any():
                rew[-1] -= penalty  # GTA encodes termination as a -100 reward (paper App. A.1)
            self.eps.append(dict(
                obs=np.asarray(ep["observations"], np.float32),
                act=np.asarray(ep["actions"], np.float32),
                next_obs=np.asarray(ep["next_observations"], np.float32),
                rew=rew.astype(np.float32),
                rtg=discounted_cumsum(rew, gamma).astype(np.float32),
            ))
        self.raw_rewards = np.concatenate([np.asarray(ep["rewards"], np.float32).reshape(-1) for ep in episodes])
        self.lengths = np.array([len(e["obs"]) for e in self.eps])
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])

    def window(self, e, s):
        ep, H = self.eps[e], self.H
        rows = np.concatenate([ep["obs"][s:s + H], ep["act"][s:s + H], ep["rew"][s:s + H, None]], axis=1)
        last = np.zeros((1, rows.shape[1]), np.float32)
        last[0, :self.obs_dim] = ep["next_obs"][s + H - 1]
        return np.concatenate([rows, last], axis=0)

    def windows(self, es, ss):
        return np.stack([self.window(int(e), int(s)) for e, s in zip(es, ss)])

    def state_std(self):
        return np.concatenate([e["obs"] for e in self.eps]).std(0) + 1e-6


def build_model(ckpt, cfg, wd, device):
    """Rebuild the GTA diffusion model and load its EMA weights (what GTA samples with)."""
    obs = np.concatenate([e["obs"] for e in wd.eps])
    act = np.concatenate([e["act"] for e in wd.eps])
    inputs = torch.from_numpy(np.concatenate([obs, act, wd.raw_rewards[:, None]], -1)).float()
    normalizer = normalizer_factory(cfg.Dataset.normalizer_type, inputs, skip_dims=[])

    T = cfg.Dataset.seq_len + 1
    data_shape = {"observations": (T, wd.obs_dim), "actions": (T, wd.act_dim), "rewards": (T, 1)}
    model = construct_diffusion_model(
        data_shape=data_shape,
        normalizer=normalizer,
        denoising_network=cfg.construct_diffusion_model.denoising_network,
        edm_config=cfg.ElucidatedDiffusion,
        cond_dim=cfg.construct_diffusion_model.denoising_network.cond_dim,
    )
    ema = torch.load(ckpt, map_location="cpu")["ema"]
    model.load_state_dict({k[len("ema_model."):]: v for k, v in ema.items() if k.startswith("ema_model.")})
    return model.to(device).eval()


def simulate(env, window, obs_dim, act_dim):
    """Put MuJoCo in each generated state, apply the generated action; return true next states and rewards."""
    sim = env.unwrapped
    nq = sim.model.nq
    T = len(window) - 1
    nxt = np.empty((T, obs_dim), np.float32)
    rew = np.empty(T, np.float32)
    for t in range(T):
        s = window[t, :obs_dim].astype(np.float64)
        sim.set_state(np.concatenate([[0.0], s[:nq - 1]]), s[nq - 1:])  # x-position is unobserved
        o, r, _, _ = sim.step(window[t, obs_dim:obs_dim + act_dim])
        nxt[t], rew[t] = o, r
    return nxt, rew


def edit_metrics(env, edited, sources, obs_dim, act_dim, state_std, desc="oracle"):
    """Paper App. D metrics for each edited window: generated vs simulator reward, dynamic MSE, shift."""
    from tqdm import tqdm
    n = len(edited)
    out = {k: np.empty(n, np.float32) for k in ["gen_r", "oracle_r", "dyn_mse", "shift"]}
    for i in tqdm(range(n), desc=desc, unit="win", leave=False):
        x = edited[i]
        true_next, true_r = simulate(env, x, obs_dim, act_dim)
        out["gen_r"][i] = x[:-1, obs_dim + act_dim].sum()
        out["oracle_r"][i] = true_r.sum()
        out["dyn_mse"][i] = (((true_next - x[1:, :obs_dim]) / state_std) ** 2).mean()
        out["shift"][i] = np.sqrt((((x[:-1, :obs_dim] - sources[i, :-1, :obs_dim]) / state_std) ** 2).mean())
    return out

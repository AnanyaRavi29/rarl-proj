"""Experiment A data: GTA datasets where each source window gets its own mu from its local support.

All conditions edit the SAME source windows with the same alpha; only the per-window mu differs.
"""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.h2.common import edit_metrics
from src.h2.edit_eval import amplified_condition, denoise_steps
from src.h2.quality import build_catalog
from src.h2.support import recon_error, recon_score

# mu per support quintile (lowest -> highest support). "adaptive" follows the mu_safe curve of the H2 study.
CONDITIONS = {
    "uniform_0.5": [0.5] * 5,
    "uniform_0.3": [0.3] * 5,
    "adaptive": [0.2, 0.25, 0.3, 0.35, 0.4],
    "inverted": [0.4, 0.35, 0.3, 0.25, 0.2],
}


def select_sources(wd, model, gamma, n_windows, sigmas, draws, batch, device, seed, out):
    """Random source windows (as upstream GTA: all start positions) + their S2 support quintile."""
    path = Path(out) / "sources.npz"
    if path.exists():
        print(f"Reusing {path}")
        return dict(np.load(path))
    cat = build_catalog(wd, gamma, max_start=10 ** 9)
    idx = np.random.default_rng(seed).choice(len(cat), n_windows, replace=False)
    src = cat.iloc[np.sort(idx)].reset_index(drop=True)
    es, ss = src.episode.values, src.start.values
    errs = recon_error(model, wd, es, ss, sigmas, draws, batch, device, seed)
    score = recon_score(errs)
    pct = (np.argsort(np.argsort(score)) + 0.5) / len(score)
    data = dict(episode=es, start=ss, rtg=src.rtg.values.astype(np.float32),
                q_return=src.q_return.values.astype(np.float32), s2_score=score.astype(np.float32),
                s_bin=np.minimum((pct * 5).astype(int), 4))
    np.savez(path, **data)
    print(f"Saved {path}: {n_windows} windows, support quintile counts {np.bincount(data['s_bin'])}")
    return data


def windows_to_transitions(windows, obs_dim, act_dim):
    """[N, H+1, D] edited windows -> flat (s, a, r, s', done) arrays (paper App. A.1)."""
    s = windows[:, :-1, :obs_dim]
    return dict(
        observations=s.reshape(-1, obs_dim),
        actions=windows[:, :-1, obs_dim:obs_dim + act_dim].reshape(-1, act_dim),
        rewards=windows[:, :-1, obs_dim + act_dim].reshape(-1),
        next_observations=windows[:, 1:, :obs_dim].reshape(-1, obs_dim),
        terminals=np.zeros(s.shape[0] * s.shape[1], np.float32),
    )


def generate_condition(name, src, wd, model, env, cfg, batch, device, seed, out, n_check=5000):
    """Edit every source window with its condition-specific mu; save transitions + a data-quality check."""
    path = Path(out) / f"{name}.npz"
    if path.exists():
        print(f"{name}: already generated ({path})")
        return
    mus = np.asarray(CONDITIONS[name], np.float32)[src["s_bin"]]
    g = cfg.SimpleDiffusionGenerator
    cond = amplified_condition(src["rtg"], g.amplify_returnscale, cfg.Dataset.reward_scale)
    n = len(mus)
    edited = np.empty((n, wd.H + 1, wd.obs_dim + wd.act_dim + 1), np.float32)

    groups = [(mu, np.flatnonzero(mus == mu)) for mu in np.unique(mus)]
    total = sum(denoise_steps(float(mu), g.num_sample_steps) * -(-len(ix) // batch) for mu, ix in groups)
    pbar = tqdm(total=total, desc=f"generate {name}", unit="step")
    for mu, ix in groups:
        torch.manual_seed(seed + int(round(mu * 1000)))
        for b in range(0, len(ix), batch):
            j = ix[b:b + batch]
            x = torch.from_numpy(wd.windows(src["episode"][j], src["start"][j])).to(device)
            c = torch.from_numpy(cond[j]).to(device)
            with torch.no_grad():
                y = model.sample_back_and_forth(samples=x, cond=c, noise_level=float(mu), temperature=g.temperature,
                                                num_sample_steps=g.num_sample_steps, disable_tqdm=True)
            edited[j] = y.cpu().numpy()
            pbar.update(denoise_steps(float(mu), g.num_sample_steps))
            pbar.set_postfix(mu=float(mu))
    pbar.close()

    # data-quality check on a random subset (simulator reward, physics error)
    chk = np.random.default_rng(seed).choice(n, min(n_check, n), replace=False)
    sources = wd.windows(src["episode"][chk], src["start"][chk])
    m = edit_metrics(env, edited[chk], sources, wd.obs_dim, wd.act_dim, wd.state_std(), desc=f"check {name}")
    m_src = edit_metrics(env, sources, sources, wd.obs_dim, wd.act_dim, wd.state_std(), desc="check sources")

    np.savez(path, mu=mus, check_idx=chk, check_src_oracle_r=m_src["oracle_r"],
             **{f"check_{k}": v for k, v in m.items()},
             **windows_to_transitions(edited, wd.obs_dim, wd.act_dim))
    print(f"{name}: saved {path} ({n * wd.H} transitions); check: oracle reward {m['oracle_r'].mean():.1f} "
          f"(sources {m_src['oracle_r'].mean():.1f}), dynamic MSE {m['dyn_mse'].mean():.4f}")

"""mu sweep: edit each selected window at several noise ratios mu and score the edits in MuJoCo."""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.h2.common import edit_metrics


def stratified_sample(pool, support_col, bins=5, per_cell=16, seed=0):
    """Equal numbers of windows from every (Q bin, S bin) cell, so trends cover the full range."""
    pool = pool.assign(
        q_bin=np.minimum((pool["Q"] * bins).astype(int), bins - 1),
        s_bin=np.minimum((pool[support_col] * bins).astype(int), bins - 1),
    )
    sel = pool.groupby(["q_bin", "s_bin"], group_keys=False).apply(
        lambda g: g.sample(min(per_cell, len(g)), random_state=seed))
    sel = sel.reset_index(drop=True)
    sel.insert(0, "window", np.arange(len(sel)))
    return sel


def amplified_condition(rtg, alpha, reward_scale):
    """Same as SimpleDiffusionGenerator.prepare_sampling_data: x alpha if positive, x (2 - alpha) if negative."""
    c = np.where(rtg > 0, rtg * alpha, rtg * (2 - alpha))
    return (c * reward_scale).astype(np.float32).reshape(-1, 1)


def denoise_steps(mu, num_steps):
    return num_steps - int(num_steps * (1 - mu))


def run_sweep(model, env, sources, cond, mus, draws, batch, temperature, num_steps,
              obs_dim, act_dim, state_std, out_dir, device, seed=0):
    """Writes one file per (mu, draw) to out_dir/sweep_parts; existing files are skipped (resumable)."""
    parts = Path(out_dir) / "sweep_parts"
    parts.mkdir(parents=True, exist_ok=True)
    todo = [(mu, d) for mu in mus for d in range(draws) if not (parts / f"mu{mu:.2f}_d{d}.npz").exists()]
    if not todo:
        print("All (mu, draw) runs already done.")
        return
    n = len(sources)
    n_batches = -(-n // batch)
    total = sum(denoise_steps(mu, num_steps) for mu, _ in todo) * n_batches
    print(f"{len(todo)} runs to do ({len(mus) * draws - len(todo)} already done), "
          f"{n} windows each, {total} batched denoising steps on {device}")

    pbar = tqdm(total=total, desc="denoising", unit="step")
    for mu, d in todo:
        pbar.set_postfix(mu=mu, draw=d)
        torch.manual_seed(seed + 1000 * d + int(round(mu * 1000)))
        edited = np.empty_like(sources)
        for b in range(0, n, batch):
            x = torch.from_numpy(sources[b:b + batch]).to(device)
            c = torch.from_numpy(cond[b:b + batch]).to(device)
            with torch.no_grad():
                y = model.sample_back_and_forth(samples=x, cond=c, noise_level=mu, temperature=temperature,
                                                num_sample_steps=num_steps, disable_tqdm=True)
            edited[b:b + batch] = y.cpu().numpy()
            pbar.update(denoise_steps(mu, num_steps))
        m = edit_metrics(env, edited, sources, obs_dim, act_dim, state_std, desc=f"oracle mu={mu} d={d}")
        np.savez(parts / f"mu{mu:.2f}_d{d}.npz", edited=edited.astype(np.float32), mu=mu, draw=d, **m)
    pbar.close()

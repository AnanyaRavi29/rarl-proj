"""Local-support metrics for GTA source windows (higher score = more support).

S1  kNN distance in normalized (s, a) space to transitions of *other* episodes (PRDC / GTA-novelty style).
S2  Reconstruction error of the trained diffusion model at fixed noise levels (unguided, one denoiser call).
S3  Exact log-likelihood via the probability-flow ODE (slow; only used to validate S2 on a small subset).
"""
import numpy as np
import torch
from tqdm import tqdm


def knn_distance(wd, es, ss, k=20, batch=4096, k_search=64):
    """Median over the window's steps of the distance to the k-th nearest transition from another episode."""
    import faiss

    sa = np.concatenate([np.concatenate([ep["obs"], ep["act"]], 1) for ep in wd.eps])
    z = np.ascontiguousarray((sa - sa.mean(0)) / (sa.std(0) + 1e-6), dtype=np.float32)
    ep_of = np.repeat(np.arange(len(wd.eps)), wd.lengths)
    index = faiss.IndexFlatL2(z.shape[1])
    index.add(z)

    ids = (wd.offsets[es] + ss)[:, None] + np.arange(wd.H)[None]
    uniq, inv = np.unique(ids.ravel(), return_inverse=True)
    kdist = np.empty(len(uniq), np.float32)
    k_max = int(wd.lengths.max()) + k
    for b in tqdm(range(0, len(uniq), batch), desc="S1 kNN", unit="batch"):
        q = uniq[b:b + batch]
        kdist[b:b + len(q)] = _kth_external(index, z, ep_of, q, k, k_search, k_max)
    return np.median(kdist[inv].reshape(ids.shape), axis=1)


def _kth_external(index, z, ep_of, q, k, k_search, k_max):
    d, i = index.search(z[q], k_search)
    d = np.where(ep_of[i] == ep_of[q][:, None], np.inf, d)
    kth = np.sort(d, axis=1)[:, k - 1]
    redo = ~np.isfinite(kth)
    if redo.any():  # fewer than k other-episode hits in the first k_search: search wider
        d2, i2 = index.search(z[q[redo]], k_max)
        d2 = np.where(ep_of[i2] == ep_of[q[redo]][:, None], np.inf, d2)
        kth[redo] = np.sort(d2, axis=1)[:, k - 1]
    return np.sqrt(np.maximum(kth, 0.0))


@torch.no_grad()
def recon_error(model, wd, es, ss, sigmas, draws=4, batch=512, device="cpu", seed=0):
    """Mean squared one-step denoising error (normalized space) per window and noise level: [N, len(sigmas)]."""
    gen = torch.Generator().manual_seed(seed)
    n = len(es)
    out = np.zeros((n, len(sigmas)), np.float32)
    pbar = tqdm(total=n * len(sigmas) * draws, desc="S2 recon", unit="win")
    for b in range(0, n, batch):
        x0 = model.normalizer.normalize(torch.from_numpy(wd.windows(es[b:b + batch], ss[b:b + batch])).to(device))
        mask = torch.ones_like(x0)
        mask[:, -1, wd.obs_dim:] = 0.0  # last row only holds the final next-state (same mask as training)
        for j, sig in enumerate(sigmas):
            err = torch.zeros(len(x0), device=device)
            for _ in range(draws):
                noise = torch.randn(x0.shape, generator=gen).to(device)
                den = model.preconditioned_network_forward(x0 + sig * noise, float(sig), cond=None)
                err += (((den - x0) ** 2) * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
                pbar.update(len(x0))
            out[b:b + len(x0), j] = (err / draws).cpu().numpy()
    pbar.close()
    return out


def recon_score(errors):
    """Combine noise levels: average z-scored log error, negated so that higher = more support."""
    le = np.log(errors)
    return -((le - le.mean(0)) / le.std(0)).mean(1)


def log_likelihood(model, wd, es, ss, batch=25, device="cpu"):
    out = []
    for b in tqdm(range(0, len(es), batch), desc="S3 log-lik", unit="batch"):
        x = torch.from_numpy(wd.windows(es[b:b + batch], ss[b:b + batch])).to(device)
        ll, _ = model.log_likelihood(x)
        out.append(ll.cpu().numpy())
    return np.concatenate(out)

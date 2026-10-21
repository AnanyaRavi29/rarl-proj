"""Per-window continuous mu and alpha for GTA, as smooth functions of support S and quality Q.

mu_i    = 0.3 + 0.4 * sigmoid(a0 + aS * S'_i + aQ * Q'_i)     in [0.3, 0.7]
alpha_i = 1.0 + 0.8 * sigmoid(c0 + cS * S'_i + cQ * Q'_i)     in [1.0, 1.8]
with S', Q' = percentile - 0.5 in [-0.5, 0.5]. Parameters x = [a0, aS, aQ, c0, cS, cQ].
"""
import math

import numpy as np
import torch

MU_LO, MU_SPAN = 0.3, 0.4
AL_LO, AL_SPAN = 1.0, 0.8


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return float(np.log(p / (1 - p)))


def default_params():
    """GTA default: mu = 0.5, alpha = 1.3 for every window, no slopes."""
    return np.array([logit((0.5 - MU_LO) / MU_SPAN), 0.0, 0.0, logit((1.3 - AL_LO) / AL_SPAN), 0.0, 0.0])


def mu_alpha(x, s, q):
    """Per-window mu and alpha from parameters x and centred percentiles s, q."""
    a0, aS, aQ, c0, cS, cQ = x
    mu = MU_LO + MU_SPAN * _sig(a0 + aS * s + aQ * q)
    alpha = AL_LO + AL_SPAN * _sig(c0 + cS * s + cQ * q)
    return mu.astype(np.float32), alpha.astype(np.float32)


@torch.no_grad()
def sample_per_window(model, samples, cond, mus, temperature=1.0, num_steps=128):
    """GTA back-and-forth editing where every window has its own mu (same maths as
    ElucidatedDiffusion.sample_back_and_forth). Each window is noised at its own level
    sigma[k0_i], k0_i = floor(num_steps * (1 - mu_i)), and only updated from that step on."""
    device = model.device
    x0 = model.normalizer.normalize(samples.to(device))
    sig = model.sample_schedule(num_steps)
    gam = torch.where((sig >= model.S_tmin) & (sig <= model.S_tmax),
                      min(model.S_churn / num_steps, math.sqrt(2) - 1), 0.0)
    k0 = torch.as_tensor(np.floor(num_steps * (1 - np.asarray(mus))).astype(np.int64), device=device)
    x = x0.clone()
    for k in range(int(k0.min()), num_steps):
        start = k0 == k
        if start.any():
            x[start] = x0[start] + sig[k] * torch.randn_like(x0[start])
        active = (k0 <= k)[:, None, None]
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
        x = torch.where(active, x_next, x)
    return model.normalizer.unnormalize(x)

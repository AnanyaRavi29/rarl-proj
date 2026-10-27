"""Support from trajectory clustering: PG-Kmeans and CAAE (Wang, Du & Hu, UAI 2026), compared with our support metrics.

The clustering unit is a whole ORIGINAL dataset trajectory (episode, 1,000 steps), as in the paper:
  PG-Kmeans: K Gaussian behaviour-cloning policies; alternate (train each policy on the transitions of its
             episodes, reassign every episode to the policy under which its actions are most likely).
  CAAE:      trajectory encoder (per-step MLP, mean-pooled) -> latent z, pulled to the nearest of K learnable
             centroids; decoder reconstructs the trajectory's actions from (state, z).

Support scores for each 31-step source window, from the trajectory clusters (higher = better supported):
  pg_ll        mean per-step log pi_k(a|s) of the window under the best-fitting cluster policy
  pg_size      share of episodes in that best-fitting PG cluster
  pg_ep_size   share of episodes in the PG cluster of the window's own episode
  caae_dist    -distance of the window's latent to the nearest trajectory centroid
  caae_rec     -action reconstruction error of the window given its own latent
  caae_size    share of episodes in the CAAE cluster nearest to the window
  caae_ep_size share of episodes in the CAAE cluster of the window's own episode
  caae_knn     -distance of the window's latent to the 20th nearest episode latent

Compared (Spearman) with S1/S2/S3/Q on the 20k pool, and with simulator physics error and mu_safe on the
400 swept windows of the H2 main study.

    python scripts/h8_cluster_support.py --k 8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
import torch
import torch.nn as nn
from scipy.stats import spearmanr

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from src.h2.common import WindowData  # noqa: E402

ENV = "halfcheetah-medium-v2"
H = 31


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


# ---------------------------------------------------------------------------------------------
class GaussPolicy(nn.Module):
    def __init__(self, sd, ad):
        super().__init__()
        self.net, self.log_std = mlp(sd, ad), nn.Parameter(torch.full((ad,), -1.0))

    def log_prob(self, s, a):
        mu, ls = self.net(s), self.log_std.clamp(-5, 2)
        return (-0.5 * ((a - mu) / ls.exp()) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)


def seq_loglik(pols, X, sd, bs=256):
    """(N, K) mean per-step log-likelihood of each sequence's actions under each policy. X: (N, T, sd+ad)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs]
            n, T = x.shape[:2]
            s, a = x[..., :sd].reshape(-1, sd), x[..., sd:].reshape(n * T, -1)
            out.append(torch.stack([p.log_prob(s, a).view(n, T).mean(1) for p in pols], 1))
    return torch.cat(out)


def pg_kmeans(E, sd, K, rounds, steps, dev, seed):
    """E: (n_episodes, T, sd+ad). Clusters episodes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ad = E.shape[-1] - sd
    n, T = E.shape[:2]
    pols = [GaussPolicy(sd, ad).to(dev) for _ in range(K)]
    opts = [torch.optim.Adam(p.parameters(), 3e-4) for p in pols]
    assign = torch.randint(0, K, (n,), generator=g).to(dev)
    hist = []
    for r in range(rounds):
        for k in range(K):
            idx = (assign == k).nonzero().squeeze(1)
            if len(idx) < 5:  # empty cluster: re-seed from a random half of the largest one
                big = torch.bincount(assign, minlength=K).argmax()
                cand = (assign == big).nonzero().squeeze(1)
                idx = cand[torch.randperm(len(cand), generator=g)[: len(cand) // 2].to(dev)]
                assign[idx] = k
            for _ in range(steps):
                e = idx[torch.randint(0, len(idx), (512,), generator=g).to(dev)]
                t = torch.randint(0, T, (512,), generator=g).to(dev)
                x = E[e, t]
                loss = -pols[k].log_prob(x[:, :sd], x[:, sd:]).mean()
                opts[k].zero_grad()
                loss.backward()
                opts[k].step()
        ll = seq_loglik(pols, E, sd, bs=32)
        new = ll.argmax(1)
        changed = (new != assign).float().mean().item()
        assign = new
        hist.append(dict(round=r, changed=changed, mean_ll=ll.max(1).values.mean().item(),
                         sizes=torch.bincount(assign, minlength=K).tolist()))
        log(f"PG-Kmeans round {r}: {changed:.1%} of episodes reassigned, mean best ll {hist[-1]['mean_ll']:.3f}, "
            f"sizes {hist[-1]['sizes']}")
        if r >= 3 and changed == 0:
            break
    return pols, assign, hist


# ---------------------------------------------------------------------------------------------
class CAAE(nn.Module):
    """Trajectory encoder (mean-pooled per-step features) + action decoder pi(a | s, z) + K centroids."""

    def __init__(self, sd, ad, K, z=16):
        super().__init__()
        self.sd = sd
        self.phi, self.rho = mlp(sd + ad, 256), mlp(256, z)
        self.dec = mlp(sd + z, ad)
        self.cent = nn.Parameter(torch.randn(K, z) * 0.5)

    def encode(self, x):  # (n, T, sd+ad) -> (n, z)
        return self.rho(self.phi(x).mean(1))

    def rec_err(self, x, z):  # per-sequence action MSE
        s, a = x[..., :self.sd], x[..., self.sd:]
        zz = z[:, None].expand(-1, x.shape[1], -1)
        return ((self.dec(torch.cat([s, zz], -1)) - a) ** 2).mean((1, 2))

    def dists(self, z):
        return torch.cdist(z, self.cent)


def train_caae(E, sd, K, steps, warm, lam, seg, dev, seed):
    torch.manual_seed(seed)
    n, T = E.shape[:2]
    m = CAAE(sd, E.shape[-1] - sd, K).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    ar = torch.arange(seg, device=dev)
    for it in range(steps):
        e = torch.randint(0, n, (64,), device=dev)
        t0 = torch.randint(0, T - seg + 1, (64,), device=dev)
        x = E[e[:, None], t0[:, None] + ar]  # random 'seg'-step piece of each trajectory
        z = m.encode(x)
        rec = m.rec_err(x, z).mean()
        if it == warm:
            with torch.no_grad():
                m.cent.copy_(m.encode(E[torch.randperm(n, device=dev)[:K]]))
        if it > warm and it % 500 == 0:  # re-seed centroids holding < 1% of episodes
            with torch.no_grad():
                ze = encode_all(m, E)
                use = torch.bincount(m.dists(ze).argmin(1), minlength=K).float() / n
                dead = (use < 0.01).nonzero().squeeze(1)
                if len(dead):
                    m.cent[dead] = ze[torch.randperm(n, device=dev)[:len(dead)]]
        att = m.dists(z).min(1).values.pow(2).mean() if it >= warm else torch.zeros((), device=dev)
        loss = rec + lam * att
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 2000 == 0 or it == steps - 1:
            log(f"CAAE step {it}: action recon {rec.item():.4f}, attract {att.item():.4f}")
    return m


def encode_all(m, X, bs=64):
    with torch.no_grad():
        return torch.cat([m.encode(X[i:i + bs]) for i in range(0, len(X), bs)])


# ---------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--pg_rounds", type=int, default=20)
    p.add_argument("--pg_steps", type=int, default=500)
    p.add_argument("--caae_steps", type=int, default=20_000)
    p.add_argument("--caae_warm", type=int, default=3_000)
    p.add_argument("--caae_lam", type=float, default=0.1)
    p.add_argument("--caae_seg", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/h8_cluster_support")
    args = p.parse_args()
    out = Path(args.out) / f"k{args.k}"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    wd = WindowData(ENV, H, 0.99)
    sd, K = wd.obs_dim, args.k
    T = int(wd.lengths.min())
    E = np.stack([np.concatenate([e["obs"][:T], e["act"][:T]], 1) for e in wd.eps])  # (1000, 1000, 23)
    mean, std = E.reshape(-1, E.shape[-1]).mean(0), E.reshape(-1, E.shape[-1]).std(0) + 1e-6
    to_t = lambda a: torch.from_numpy((a - mean) / std).float().to(dev)  # noqa: E731
    E_t = to_t(E)

    pool = pd.read_csv("outputs/h2/pool.csv")
    W = wd.windows(pool.episode.values, pool.start.values)[:, :H, :sd + wd.act_dim]
    W_t = to_t(W)
    ep = torch.as_tensor(pool.episode.values, device=dev)
    log(f"{len(E)} trajectories x {T} steps; pool {len(pool)} windows; K={K}; device {dev}")

    t0 = time.time()
    pols, assign, hist = pg_kmeans(E_t, sd, K, args.pg_rounds, args.pg_steps, dev, args.seed)
    t_pg = time.time() - t0
    size_pg = torch.bincount(assign, minlength=K).float() / len(assign)
    ll = seq_loglik(pols, W_t, sd)
    pool["pg_cluster"] = ll.argmax(1).cpu().numpy()
    pool["pg_ll"] = ll.max(1).values.cpu().numpy()
    pool["pg_size"] = size_pg[ll.argmax(1)].cpu().numpy()
    pool["pg_ep_size"] = size_pg[assign[ep]].cpu().numpy()

    t0 = time.time()
    m = train_caae(E_t, sd, K, args.caae_steps, args.caae_warm, args.caae_lam, args.caae_seg, dev, args.seed)
    t_caae = time.time() - t0
    ze = encode_all(m, E_t)
    ca_assign = m.dists(ze).argmin(1)
    size_ca = torch.bincount(ca_assign, minlength=K).float() / len(ze)
    zw = encode_all(m, W_t, bs=4096)
    with torch.no_grad():
        dw = m.dists(zw)
        recw = torch.cat([m.rec_err(W_t[i:i + 4096], zw[i:i + 4096]) for i in range(0, len(W_t), 4096)])
    pool["caae_cluster"] = dw.argmin(1).cpu().numpy()
    pool["caae_dist"] = -dw.min(1).values.cpu().numpy()
    pool["caae_rec"] = -recw.cpu().numpy()
    pool["caae_size"] = size_ca[dw.argmin(1)].cpu().numpy()
    pool["caae_ep_size"] = size_ca[ca_assign[ep]].cpu().numpy()
    pool["caae_knn"] = -torch.cdist(zw, ze).topk(20, largest=False).values[:, -1].cpu().numpy()

    # ---- correlations ----
    new = ["pg_ll", "pg_size", "pg_ep_size", "caae_dist", "caae_rec", "caae_size", "caae_ep_size", "caae_knn"]
    ref = {"S1 (kNN)": "S1", "S2 (recon.)": "S2", "S3 (exact loglik)": "s3_loglik", "Q (quality)": "Q"}
    rows = []
    for c in new:
        r = {"metric": c}
        for name, col in ref.items():
            ok = pool[col].notna()
            r[name] = spearmanr(pool.loc[ok, c], pool.loc[ok, col])[0]
        rows.append(r)
    corr_pool = pd.DataFrame(rows)

    sw = pd.read_csv("outputs/h2/sweep.csv")
    sw = sw[sw.mu <= 0.5].groupby(["window", "mu"]).dyn_mse.mean().unstack()
    best = pd.read_csv("outputs/h2/best_mu.csv").set_index("window")
    sel = pd.read_csv("outputs/h2/selected.csv")
    sel = sel.merge(pool[["episode", "start"] + new], on=["episode", "start"], how="left")
    sel["dyn_mean"] = sw.mean(1).reindex(sel.window).values
    sel["dyn_0.3"] = sw[0.3].reindex(sel.window).values
    sel["mu_safe"] = best.mu_safe.reindex(sel.window).values
    rows = []
    for c in new + ["S1", "S2"]:
        rows.append({"metric": c,
                     "physics error (mean, mu<=0.5)": spearmanr(sel[c], sel.dyn_mean)[0],
                     "physics error (mu=0.3)": spearmanr(sel[c], sel["dyn_0.3"])[0],
                     "mu_safe": spearmanr(sel[c], sel.mu_safe)[0]})
    corr_sel = pd.DataFrame(rows)

    def fmt(d):
        head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns)
        body = ["| " + " | ".join(v if isinstance(v, str) else f"{v:+.2f}" for v in r) + " |" for r in d.values]
        return "\n".join([head] + body)

    pool.to_csv(out / "pool_scores.csv", index=False)
    md = [f"# Trajectory-cluster support (PG-Kmeans, CAAE), K = {K}", "",
          f"Clustered the {len(E)} original trajectories ({T} steps each). PG-Kmeans {len(hist)} rounds "
          f"({t_pg/60:.1f} min), CAAE {args.caae_steps} steps ({t_caae/60:.1f} min).",
          f"PG cluster sizes (episodes): {hist[-1]['sizes']}; CAAE cluster sizes: {torch.bincount(ca_assign, minlength=K).tolist()}",
          "",
          "## Agreement with existing support metrics (20k pool; S3 on 100 windows)", "", fmt(corr_pool), "",
          "## Prediction of edit damage (400 swept windows)",
          "Good support metric: negative with physics error, positive with mu_safe.", "", fmt(corr_sel)]
    (out / "summary.md").write_text("\n".join(md))
    json.dump(dict(args=vars(args), pg_history=hist, corr_pool=corr_pool.to_dict("records"),
                   corr_sel=corr_sel.to_dict("records")), open(out / "results.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()

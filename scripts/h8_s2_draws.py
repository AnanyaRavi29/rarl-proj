"""Is the S2-S3 gap sampling noise? Recompute S2 on the 100 S3 windows with 4 independent sets of 4 draws
(= 16 draws), standardised with the pool statistics, and correlate each version with S3.

    python scripts/h8_s2_draws.py
"""
import json

import numpy as np
import pandas as pd
import pyrootutils
from scipy.stats import spearmanr

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from src.h2.common import WindowData, build_model, load_cfg, pick_device  # noqa: E402
from src.h2.support import recon_error  # noqa: E402

CKPT = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"
SIGMAS = [0.16, 0.5, 2.4]
SETS, DRAWS = 4, 4

pool = pd.read_csv("outputs/h2/pool.csv")
s3 = pool.dropna(subset=["s3_loglik"]).reset_index(drop=True)
pool_le = np.log(pool[[f"s2_err_sigma{s:g}" for s in SIGMAS]].values)
mu, sd = pool_le.mean(0), pool_le.std(0)


def score(err):
    return -((np.log(err) - mu) / sd).mean(1)


cfg = load_cfg(CKPT)
wd = WindowData("halfcheetah-medium-v2", cfg.Dataset.seq_len, cfg.Dataset.gamma)
device = pick_device("auto")
model = build_model(CKPT, cfg, wd, device)
es, ss = s3.episode.values, s3.start.values

errs = [recon_error(model, wd, es, ss, SIGMAS, DRAWS, 100, device, seed=100 + k) for k in range(SETS)]
scores4 = [score(e) for e in errs]
score16 = score(np.mean(errs, 0))

y = s3.s3_loglik.values
rng = np.random.default_rng(0)


def ci(x):
    b = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    return np.percentile(b, [2.5, 97.5])


rows = [("original S2 (4 draws, pool run)", s3.s2_score.values)]
rows += [(f"new 4-draw set {k + 1}", s) for k, s in enumerate(scores4)]
rows += [("16 draws (all 4 sets)", score16)]
res = {name: dict(rho=float(spearmanr(x, y)[0]), ci=[float(v) for v in ci(x)]) for name, x in rows}
for j, sig in enumerate(SIGMAS):  # each noise level alone, 16 draws
    x = -np.log(np.mean(errs, 0)[:, j])
    res[f"16 draws, sigma={sig:g} only"] = dict(rho=float(spearmanr(x, y)[0]), ci=[float(v) for v in ci(x)])

pairs = [spearmanr(scores4[a], scores4[b])[0] for a in range(SETS) for b in range(a + 1, SETS)]
res["repeatability: rho between independent 4-draw sets (mean, min)"] = [float(np.mean(pairs)), float(np.min(pairs))]
res["repeatability: rho original vs 16-draw"] = float(spearmanr(s3.s2_score, score16)[0])

for k, v in res.items():
    print(f"{k:60s} {v}")
json.dump(res, open("outputs/h8_cluster_support/s2_draws.json", "w"), indent=1)

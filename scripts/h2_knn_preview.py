"""Quick look at S1 (kNN support) on a few hundred windows, without the diffusion model.

    nice -n 10 python scripts/h2_knn_preview.py --n 500 --threads 4
"""
import argparse
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.h2.common import WindowData
from src.h2.quality import build_catalog
from src.h2.support import knn_distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="halfcheetah-medium-v2")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="outputs/h2_preview")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    faiss.omp_set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wd = WindowData(args.env, H=31, gamma=0.99)
    cat = build_catalog(wd, 0.99)
    pool = cat.sample(args.n, random_state=args.seed).reset_index(drop=True)
    pool["s1_dist"] = knn_distance(wd, pool.episode.values, pool.start.values, batch=1024)
    pool["Q"] = pool.q_return.rank(pct=True)
    pool["S1"] = (-pool.s1_dist).rank(pct=True)
    pool.to_csv(out / "knn_preview.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(pool.s1_dist, bins=40)
    ax[0].set_xlabel("distance to 20th nearest step from other runs")
    ax[0].set_title("S1 distances (lower = more support)")
    ax[1].scatter(pool.Q, pool.S1, s=10, alpha=0.6)
    ax[1].set_xlabel("quality Q percentile")
    ax[1].set_ylabel("support S1 percentile")
    ax[1].set_title(f"Quality vs support: Spearman {spearmanr(pool.Q, pool.S1)[0]:.3f}")
    ax[2].scatter(pool.start, pool.s1_dist, s=10, alpha=0.6, c=pool.Q, cmap="viridis")
    ax[2].set_xlabel("window start step in episode")
    ax[2].set_ylabel("S1 distance")
    ax[2].set_title("Support along the episode (colour = quality)")
    fig.tight_layout()
    fig.savefig(out / "knn_preview.png", dpi=120)
    p10, p50, p90 = np.percentile(pool.s1_dist, [10, 50, 90])
    print(f"S1 distance p10/p50/p90 = {p10:.3f} / {p50:.3f} / {p90:.3f}")
    print(f"Spearman(Q, S1) = {spearmanr(pool.Q, pool.S1)[0]:.3f}")
    print(f"Saved {out / 'knn_preview.png'}")


if __name__ == "__main__":
    main()

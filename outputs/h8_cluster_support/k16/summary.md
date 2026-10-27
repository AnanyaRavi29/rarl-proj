# Trajectory-cluster support (PG-Kmeans, CAAE), K = 16

Clustered the 1000 original trajectories (1000 steps each). PG-Kmeans 11 rounds (3.1 min), CAAE 20000 steps (0.5 min).
PG cluster sizes (episodes): [40, 28, 90, 24, 6, 6, 577, 11, 23, 52, 17, 46, 14, 9, 39, 18]; CAAE cluster sizes: [51, 56, 69, 8, 29, 14, 112, 8, 192, 25, 13, 70, 39, 302, 10, 2]

## Agreement with existing support metrics (20k pool; S3 on 100 windows)

| metric | S1 (kNN) | S2 (recon.) | S3 (exact loglik) | Q (quality) |
|---|---|---|---|---|
| pg_ll | +0.78 | +0.73 | +0.77 | +0.41 |
| pg_size | -0.26 | -0.23 | -0.18 | -0.09 |
| pg_ep_size | +0.04 | +0.04 | +0.12 | +0.04 |
| caae_dist | +0.21 | +0.19 | +0.17 | +0.08 |
| caae_rec | +0.74 | +0.72 | +0.74 | +0.38 |
| caae_size | +0.06 | +0.04 | +0.05 | +0.06 |
| caae_ep_size | +0.02 | +0.03 | -0.07 | +0.02 |
| caae_knn | +0.21 | +0.19 | +0.13 | +0.08 |

## Prediction of edit damage (400 swept windows)
Good support metric: negative with physics error, positive with mu_safe.

| metric | physics error (mean, mu<=0.5) | physics error (mu=0.3) | mu_safe |
|---|---|---|---|
| pg_ll | -0.32 | -0.25 | +0.27 |
| pg_size | +0.09 | +0.15 | -0.11 |
| pg_ep_size | -0.08 | -0.06 | +0.05 |
| caae_dist | +0.02 | -0.04 | -0.08 |
| caae_rec | -0.29 | -0.20 | +0.24 |
| caae_size | -0.01 | -0.04 | +0.02 |
| caae_ep_size | -0.07 | -0.07 | +0.06 |
| caae_knn | -0.02 | -0.06 | -0.04 |
| S1 | -0.41 | -0.32 | +0.35 |
| S2 | -0.39 | -0.34 | +0.36 |
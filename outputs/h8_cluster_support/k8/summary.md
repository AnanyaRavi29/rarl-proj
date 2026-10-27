# Trajectory-cluster support (PG-Kmeans, CAAE), K = 8

Clustered the 1000 original trajectories (1000 steps each). PG-Kmeans 20 rounds (6.1 min), CAAE 20000 steps (1.1 min).
PG cluster sizes (episodes): [44, 203, 124, 484, 43, 41, 50, 11]; CAAE cluster sizes: [140, 174, 16, 63, 53, 406, 130, 18]

## Agreement with existing support metrics (20k pool; S3 on 100 windows)

| metric | S1 (kNN) | S2 (recon.) | S3 (exact loglik) | Q (quality) |
|---|---|---|---|---|
| pg_ll | +0.79 | +0.75 | +0.76 | +0.41 |
| pg_size | -0.10 | -0.08 | +0.07 | -0.02 |
| pg_ep_size | +0.06 | +0.05 | +0.28 | +0.07 |
| caae_dist | +0.21 | +0.19 | +0.16 | +0.09 |
| caae_rec | +0.74 | +0.72 | +0.73 | +0.38 |
| caae_size | +0.01 | +0.00 | +0.02 | +0.03 |
| caae_ep_size | +0.02 | +0.02 | -0.08 | +0.01 |
| caae_knn | +0.22 | +0.19 | +0.18 | +0.11 |

## Prediction of edit damage (400 swept windows)
Good support metric: negative with physics error, positive with mu_safe.

| metric | physics error (mean, mu<=0.5) | physics error (mu=0.3) | mu_safe |
|---|---|---|---|
| pg_ll | -0.32 | -0.25 | +0.27 |
| pg_size | +0.08 | +0.11 | -0.01 |
| pg_ep_size | -0.11 | -0.03 | +0.07 |
| caae_dist | -0.11 | -0.13 | +0.08 |
| caae_rec | -0.30 | -0.20 | +0.24 |
| caae_size | -0.01 | -0.01 | +0.03 |
| caae_ep_size | -0.07 | -0.14 | +0.09 |
| caae_knn | -0.10 | -0.14 | +0.09 |
| S1 | -0.41 | -0.32 | +0.35 |
| S2 | -0.39 | -0.34 | +0.36 |
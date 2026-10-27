# Acceleration edits (first steps pinned to the source), TD3+BC (JAX), score = mean of evals 960k-1M

| condition | mean | std | seeds | unpinned GTA, same alpha |
|---|---|---|---|---|
| accel_k6_alpha_1.5 | 50.21 | 1.64 | 3 | 58.10 (diff -7.89, p = 0.004) |
| accel_k6_alpha_1.8 | 38.50 | 2.49 | 3 | 56.28 (diff -17.77, p = 0.003) |

References: GTA default (mu 0.5, alpha 1.3) 58.5; best tuned global 59.05; +5 target 63.5.
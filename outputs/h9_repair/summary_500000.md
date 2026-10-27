# Repaired GTA data, TD3+BC (JAX), score = mean of evals at 460000..500000 steps

| condition | mean | std | seeds | unrepaired (same alpha, 1M steps) |
|---|---|---|---|---|
| repaired alpha 1.5 | 54.31 | 0.00 | 1 | 58.10 (diff -3.79, p = nan) |
| repaired alpha 1.8 | 52.68 | 0.00 | 1 | 56.28 (diff -3.59, p = nan) |
| repaired alpha 2 | 51.10 | 0.00 | 1 | 52.61 (diff -1.50, p = nan) |

References: GTA default (mu 0.5, alpha 1.3) 58.51-58.66; best tuned global 59.05; oracle-weighted alpha 1.5 59.56.
# TWGA Gate 0: oracle trust weights (true physics error), JAX TD3+BC, seeds 0-2

Weight for generated transitions: exp(-error / 0.0085); original transitions weight 1.

Reference: unweighted GTA alpha 1.3 = 58.42; best tuned global (mu 0.565, alpha 1.356) = 59.05.

| alpha | unweighted | oracle-weighted | difference | p |
|---|---|---|---|---|
| 1.5 | 58.10 ± 1.10 | 59.56 ± 0.52 | +1.46 | 0.133 |
| 1.8 | 56.28 ± 0.93 | 56.98 ± 0.11 | +0.70 | 0.321 |
| 2 | 52.61 ± 0.77 | 53.34 ± 0.69 | +0.73 | 0.289 |

**Gate 0 PASSED:** best oracle-weighted score 59.56 vs 59.05 needed (beats the best tuned global setting).
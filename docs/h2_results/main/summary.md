# H2: does the best editing strength mu depend on quality Q and local support S?

Support metric: **S2**, windows: 400, mus: [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5], draws: 2, dynamic-MSE threshold: 0.0085

Hypothesis: mu* rises with S and falls with Q. Decision rule: S coefficient CI > 0, Q coefficient CI < 0, and mu*(low Q/high S) - mu*(high Q/low S) >= 0.1.

Outcomes per window: `mu_star` = highest reward gain with dynamic MSE <= threshold; `mu90` = smallest mu reaching 90% of the max gain; `mu_safe` = largest mu before dynamic MSE first exceeds the threshold (added after the first main-run analysis, so exploratory there).

## mu_star: PARTLY SUPPORTED (S effect only, small)

- Spearman with Q: -0.020, with S: 0.182
- Regression (n=400), effect across the full percentile range:
  - Q: -0.001  CI95 [-0.025, +0.022]  permutation p=0.921
  - S: +0.059  CI95 [+0.030, +0.090]  permutation p=0.000
  - QxS: -0.053  CI95 [-0.166, +0.051]
- Quadrant means: {'high Q / high S': 0.482, 'high Q / low S': 0.451, 'low Q / high S': 0.485, 'low Q / low S': 0.464}
- Extreme-quadrant difference: +0.034

## mu90: PARTLY SUPPORTED (Q effect only, small)

- Spearman with Q: -0.392, with S: -0.024
- Regression (n=400), effect across the full percentile range:
  - Q: -0.048  CI95 [-0.063, -0.034]  permutation p=0.000
  - S: -0.002  CI95 [-0.014, +0.009]  permutation p=0.776
  - QxS: -0.027  CI95 [-0.085, +0.022]
- Quadrant means: {'high Q / high S': 0.459, 'high Q / low S': 0.464, 'low Q / high S': 0.484, 'low Q / low S': 0.484}
- Extreme-quadrant difference: +0.020

## mu_safe: PARTLY SUPPORTED (S effect only)

- Spearman with Q: -0.004, with S: 0.364
- Regression (n=400), effect across the full percentile range:
  - Q: -0.013  CI95 [-0.061, +0.036]  permutation p=0.585
  - S: +0.183  CI95 [+0.139, +0.225]  permutation p=0.000
  - QxS: +0.013  CI95 [-0.143, +0.171]
- Quadrant means: {'high Q / high S': 0.315, 'high Q / low S': 0.213, 'low Q / high S': 0.322, 'low Q / low S': 0.256}
- Extreme-quadrant difference: +0.109

## Correlation profile (Spearman, per mu)

| mu | support vs physics error | quality vs physics error | support vs gain | quality vs gain |
|---|---|---|---|---|
| 0.1 | -0.285 | +0.067 | +0.106 | +0.127 |
| 0.2 | -0.356 | +0.006 | +0.114 | +0.171 |
| 0.3 | -0.341 | +0.009 | +0.119 | +0.007 |
| 0.35 | -0.276 | +0.085 | +0.138 | -0.113 |
| 0.4 | -0.241 | +0.044 | +0.086 | -0.249 |
| 0.45 | -0.176 | -0.143 | +0.058 | -0.507 |
| 0.5 | -0.128 | -0.069 | +0.064 | -0.648 |

## Settings to keep in mind
- Guidance uses the config's `temperature: 1.0` (w = 0), not the paper's w = 2.
- The upstream network drops the return condition at random (p = 0.25) even while sampling; kept as-is.

Figures: `figures/heatmaps.png`, `figures/quadrant_curves.png`, `figures/best_mu_vs_Q_S.png`, `figures/best_mu_map.png`, `figures/ceiling_check.png`, `figures/support_vs_physics.png`, `figures/correlation_summary.png`, `pool_overview.png`.
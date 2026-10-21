# H2: does the best editing strength mu depend on quality Q and local support S?

Support metric: **S1**, windows: 184, mus: [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75], draws: 2, dynamic-MSE threshold: 0.0085

Hypothesis: mu* rises with S and falls with Q. Decision rule: S coefficient CI > 0, Q coefficient CI < 0, and mu*(low Q/high S) - mu*(high Q/low S) >= 0.1.

## mu_star: PARTLY SUPPORTED (Q effect only, small)

- Spearman with Q: -0.263, with S: -0.086
- Regression (n=184), effect across the full percentile range:
  - Q: -0.151  CI95 [-0.221, -0.077]  permutation p=0.000
  - S: -0.006  CI95 [-0.080, +0.073]  permutation p=0.881
  - QxS: -0.085  CI95 [-0.346, +0.167]
- Quadrant means: {'high Q / high S': 0.573, 'high Q / low S': 0.582, 'low Q / high S': 0.649, 'low Q / low S': 0.654}
- Extreme-quadrant difference: +0.067

## mu90: PARTLY SUPPORTED (Q effect only, small)

- Spearman with Q: -0.377, with S: -0.174
- Regression (n=183), effect across the full percentile range:
  - Q: -0.128  CI95 [-0.187, -0.072]  permutation p=0.000
  - S: -0.028  CI95 [-0.087, +0.033]  permutation p=0.390
  - QxS: +0.089  CI95 [-0.121, +0.295]
- Quadrant means: {'high Q / high S': 0.521, 'high Q / low S': 0.502, 'low Q / high S': 0.565, 'low Q / low S': 0.58}
- Extreme-quadrant difference: +0.063

## Settings to keep in mind
- Guidance uses the config's `temperature: 1.0` (w = 0), not the paper's w = 2.
- The upstream network drops the return condition at random (p = 0.25) even while sampling; kept as-is.

Figures: `figures/heatmaps.png`, `figures/quadrant_curves.png`, `figures/best_mu_vs_Q_S.png`, `figures/best_mu_map.png`, `figures/ceiling_check.png`, `pool_overview.png`.
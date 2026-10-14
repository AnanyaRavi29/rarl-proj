# Experiment A: does support-adaptive mu improve TD3+BC?

Same 161,291 source windows in every GTA condition; only the per-window mu differs. TD3+BC (CORL), batch 1024, 1M steps; final score = mean of the last 5 evaluations (10 episodes each).

**Verdict (mean mu 0.3):** SUPPORTED: support-adaptive mu beats both same-average uniform mu and the inverted mapping

**Verdict (mean mu 0.5, wide):** NOT SUPPORTED: adaptive_wide does not significantly beat both uniform_0.5 and inverted_wide

## Final D4RL scores

| condition | mean | std | seeds |
|---|---|---|---|
| none | 48.72 | 0.06 | 3 |
| uniform_0.3 | 51.51 | 0.36 | 3 |
| adaptive | 54.91 | 0.41 | 3 |
| inverted | 51.81 | 0.60 | 3 |
| uniform_0.5 | 59.31 | 0.30 | 3 |
| adaptive_wide | 58.60 | 0.65 | 3 |
| inverted_wide | 58.81 | 0.70 | 3 |

## Adaptive vs others (Welch t-test)

- adaptive_vs_uniform_0.3: +3.40 points, p = 0.000
- adaptive_vs_inverted: +3.10 points, p = 0.003
- adaptive_vs_uniform_0.5: -4.40 points, p = 0.000
- adaptive_vs_none: +6.19 points, p = 0.001
- adaptive_wide_vs_uniform_0.5: -0.71 points, p = 0.195
- adaptive_wide_vs_inverted_wide: -0.21 points, p = 0.722

## Generated data quality (5,000-window check per condition)

| condition | mean mu | relative reward gain | dynamic MSE | share over 0.0085 | state shift |
|---|---|---|---|---|---|
| uniform_0.3 | 0.300 | 0.057 | 0.0096 | 0.36 | 0.24 |
| adaptive | 0.300 | 0.079 | 0.0090 | 0.34 | 0.26 |
| inverted | 0.300 | 0.083 | 0.0091 | 0.33 | 0.27 |
| uniform_0.5 | 0.500 | 0.327 | 0.0076 | 0.25 | 0.88 |
| adaptive_wide | 0.500 | 0.220 | 0.0092 | 0.33 | 0.81 |
| inverted_wide | 0.500 | 0.319 | 0.0075 | 0.24 | 0.85 |

Figure: `figures/rl_results.png`
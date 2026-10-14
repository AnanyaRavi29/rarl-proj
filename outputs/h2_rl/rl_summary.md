# Experiment A: does support-adaptive mu improve TD3+BC?

Same 161,291 source windows in every GTA condition; only the per-window mu differs. TD3+BC (CORL), batch 1024, 1M steps; final score = mean of the last 5 evaluations (10 episodes each).

**Verdict:** SUPPORTED: support-adaptive mu beats both same-average uniform mu and the inverted mapping

## Final D4RL scores

| condition | mean | std | seeds |
|---|---|---|---|
| none | 48.65 | 0.18 | 3 |
| uniform_0.5 | 58.67 | 0.45 | 3 |
| uniform_0.3 | 51.02 | 0.66 | 3 |
| adaptive | 53.53 | 0.37 | 3 |
| inverted | 51.72 | 0.49 | 3 |

## Adaptive vs others (Welch t-test)

- adaptive_vs_uniform_0.3: +2.51 points, p = 0.009
- adaptive_vs_inverted: +1.81 points, p = 0.009
- adaptive_vs_uniform_0.5: -5.14 points, p = 0.000
- adaptive_vs_none: +4.88 points, p = 0.000

## Generated data quality (5,000-window check per condition)

| condition | mean mu | relative reward gain | dynamic MSE | share over 0.0085 | state shift |
|---|---|---|---|---|---|
| uniform_0.5 | 0.500 | 0.327 | 0.0076 | 0.25 | 0.88 |
| uniform_0.3 | 0.300 | 0.057 | 0.0096 | 0.36 | 0.24 |
| adaptive | 0.300 | 0.079 | 0.0090 | 0.34 | 0.26 |
| inverted | 0.300 | 0.083 | 0.0091 | 0.33 | 0.27 |

Figure: `figures/rl_results.png`
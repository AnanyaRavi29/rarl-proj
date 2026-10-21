# Alpha experiment (mu = 0.5, 1M generated transitions, JAX TD3+BC, 3 seeds)

| condition | mean | std | seeds |
|---|---|---|---|
| alpha_1 | 49.14 | 0.14 | 3 |
| alpha_1.1 | 51.35 | 0.14 | 3 |
| alpha_1.3 | 58.57 | 0.38 | 8 |
| alpha_1.5 | 58.10 | 1.10 | 3 |
| alpha_1.8 | 56.28 | 0.93 | 3 |
| alpha_2 | 52.61 | 0.77 | 3 |
| alpha_up | 59.53 | 0.51 | 8 |
| alpha_down | 59.18 | 0.62 | 8 |

**Stage 1 best global alpha: 1.3**
- alpha_up vs alpha_1.3: +0.96 (Welch p = 0.001)
- alpha_down vs alpha_1.3: +0.61 (Welch p = 0.035)
- alpha_up vs alpha_down: +0.35 (Welch p = 0.243)

## Stage 3: support vs random alpha mix (fresh seeds 10-14, mean alpha 1.3)

| condition | mean | std | seeds |
|---|---|---|---|
| alpha_1.3 | 58.66 | 0.44 | 5 |
| alpha_down | 59.40 | 0.58 | 5 |
| alpha_random | 58.69 | 1.72 | 5 |
| alpha_up | 59.52 | 0.58 | 5 |
- alpha_up vs alpha_random: +0.83 (Welch p = 0.353)
- alpha_up vs alpha_down: +0.12 (Welch p = 0.743)
- alpha_random vs alpha_1.3: +0.04 (Welch p = 0.965)
- alpha_up vs alpha_1.3: +0.87 (Welch p = 0.031)

Reading: alpha_up > alpha_random means support matters; alpha_random > alpha_1.3 with alpha_up ~ alpha_random means mixing alpha matters.
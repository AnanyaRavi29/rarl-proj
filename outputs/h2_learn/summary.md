# Learned continuous mu and alpha (CMA-ES)

**global search** (13 candidates, 500k generated transitions, 2 seeds): best 58.60 at mean mu 0.565, mean alpha 1.356, x = [0.66958, 0.0, 0.0, -0.22029, 0.0, 0.0]

**learned search** (43 candidates, 500k generated transitions, 2 seeds): best 59.93 at mean mu 0.592, mean alpha 1.277, x = [1.16137, -1.4111, 3.29415, -1.08797, 6.1504, -6.95764]

## Confirmation (1M generated transitions, 5 fresh seeds)

| setting | score | seeds | mean mu (sd) | mean alpha (sd) | edited reward | dyn MSE |
|---|---|---|---|---|---|---|
| default | 58.51 ± 0.26 | 58.7, 58.7, 58.5, 58.6, 58.1 | 0.500 (0.000) | 1.300 (0.000) | 196.6 | 0.0075 |
| best_global | 59.05 ± 0.15 | 59.3, 59.1, 58.9, 59.0, 59.1 | 0.565 (0.000) | 1.356 (0.000) | 210.2 | 0.0066 |
| best_learned | 58.76 ± 0.52 | 58.2, 58.2, 59.3, 59.0, 59.1 | 0.592 (0.059) | 1.277 (0.232) | 191.6 | 0.0091 |
- best_learned vs best_global: -0.29 (Welch p = 0.291)
- best_learned vs default: +0.25 (Welch p = 0.372)
- best_global vs default: +0.54 (Welch p = 0.006)

Decision rule: the learned map helps if best_learned beats best_global (p < 0.05).
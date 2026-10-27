# H7 / T1: does the bound explain GTA's alpha optimum?

B_c(alpha) = [eta(alpha) - c * d(alpha)] / (1 - gamma); eta = true reward gain per generated transition, d = mean next-state error (dataset-std units), c = gamma * L_V.

| alpha | eta (true reward gain / step) | d (next-state error) | measured RL score |
|---|---|---|---|
| 1 | +0.136 | 0.1929 | 49.14 |
| 1.1 | +0.440 | 0.1872 | 51.35 |
| 1.3 | +1.511 | 0.1837 | 58.42 |
| 1.5 | +2.621 | 0.2298 | 58.10 |
| 1.8 | +3.578 | 0.3467 | 56.28 |
| 2 | +3.858 | 0.4546 | 52.61 |

- Measured RL optimum: alpha = 1.3.
- Worst-case theoretical constant c = gamma * R_max / (1 - gamma) = 824 (R_max = 8.33) -> bound maximiser alpha = 1.3.
- Range of c whose maximiser lies in [1.3, 1.5]: 8.31 to 1e+03.
- Best rank agreement between B_c and the RL curve: Spearman 0.89 at c = 12.8 (maximiser alpha = 1.5).

**T1 PASSED:** a range of c reproduces the interior optimum and the curve shape (explanatory only: c fitted on one dataset).
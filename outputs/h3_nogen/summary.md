# H3: is the diffusion model necessary? (JAX TD3+BC, 1M steps, 3 seeds)

Extra data: 5M transitions resampled from the original data (no generation); gta_ref = GTA mu 0.5, alpha 1.3 (1M generated).

| condition | mean | std | seeds |
|---|---|---|---|
| none | 48.77 | 0.17 | 3 |
| rw_b0.5 | 49.27 | 0.08 | 3 |
| rw_b1 | 49.66 | 0.43 | 3 |
| rw_b2 | 50.24 | 0.29 | 3 |
| top25 | 48.53 | 0.22 | 3 |
| top10 | 48.85 | 0.24 | 3 |
| gta_ref | 58.18 | 0.19 | 3 |

- rw_b0.5: vs none +0.50 (p = 0.022), vs GTA -8.91 (p = 0.000); recovers 5% of GTA's gain
- rw_b1: vs none +0.89 (p = 0.056), vs GTA -8.52 (p = 0.000); recovers 9% of GTA's gain
- rw_b2: vs none +1.47 (p = 0.004), vs GTA -7.94 (p = 0.000); recovers 16% of GTA's gain
- top25: vs none -0.24 (p = 0.216), vs GTA -9.65 (p = 0.000); recovers -3% of GTA's gain
- top10: vs none +0.08 (p = 0.664), vs GTA -9.33 (p = 0.000); recovers 1% of GTA's gain
# H5: VAE-edited actions + dynamics-ensemble rollouts (1,000 windows)

Source windows: true reward 149.6. GTA reference: +32%, dynamic MSE 0.0075. H4 (VAE states): +28%, 0.046.

| actions | alpha | lam | true reward | gain | model reward - true | dynamic MSE | shift |
|---|---|---|---|---|---|---|---|
| original | - | - | 134.5 | -10.1% | +0.13 | 0.0349 | 0.68 |
| VAE-edited | 1.1 | 0.1 | 131.7 | -12.0% | +0.03 | 0.0380 | 0.74 |
| VAE-edited | 1.1 | 1.0 | 132.2 | -11.7% | -0.00 | 0.0379 | 0.73 |
| VAE-edited | 1.3 | 0.1 | 122.1 | -18.4% | +0.24 | 0.0325 | 0.81 |
| VAE-edited | 1.3 | 1.0 | 127.8 | -14.6% | +0.13 | 0.0346 | 0.75 |
| VAE-edited | 1.5 | 0.1 | 111.5 | -25.5% | +0.51 | 0.0351 | 0.93 |
| VAE-edited | 1.5 | 1.0 | 120.0 | -19.8% | +0.33 | 0.0304 | 0.81 |
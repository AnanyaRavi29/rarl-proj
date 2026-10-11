# GTA Reproduction Notes

## Completed

### Environment
- Python 3.9 Conda environment
- PyTorch 2.1 with CUDA
- NVIDIA A100 GPUs verified
- MuJoCo 2.1 installed for legacy mujoco-py
- D4RL successfully configured

### Dataset
Dataset:

`halfcheetah-medium-v2`

D4RL shapes:

- observations: `(1000000, 17)`
- actions: `(1000000, 6)`
- rewards: `(1000000,)`

Preprocessed GTA dataset:

`data/halfcheetah-medium-v2.pkl`

The generated dataset is intentionally excluded from Git.

### Diffusion Model

Configuration:

`configs/gta.yaml`

Important settings:

- MixerUnet
- horizon: 31
- condition dropout: 0.25
- train batch size: 256
- learning rate: 3e-4
- training steps: 1,000,000
- diffusion sampling steps: 128
- sigma_data: 1.0
- S_churn: 80
- S_tmin: 0.05
- S_tmax: 50
- S_noise: 1.003

Model size observed during training:

`29,628,949` trainable parameters.

Diffusion training completed successfully.

### Experiment Tracking
Weights & Biases was configured for experiment tracking.

### Server Execution
Long training jobs were run with `nohup`.

## Next

1. Identify and validate the final PyTorch diffusion checkpoint.
2. Generate GTA trajectories using back-and-forth sampling.
3. Train TD3+BC on the GTA-augmented dataset.
4. Implement GTA diffusion in JAX/Flax.
5. Implement or integrate JAX TD3+BC.
6. Compare PyTorch and JAX results.

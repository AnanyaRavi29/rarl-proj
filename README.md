# GTA Reproduction + JAX Port

Reproduction and extension of:

**Generative Trajectory Augmentation with Guidance for Offline Reinforcement Learning (GTA)**  
NeurIPS 2024

Original implementation:  
https://github.com/Jaewoopudding/GTA

This repository contains our reproduction of the original PyTorch GTA
implementation and an ongoing JAX reimplementation.

## Current Status

### Original PyTorch GTA

Successfully reproduced the diffusion-training pipeline for:

- Environment: `halfcheetah-medium-v2`
- Dataset: D4RL
- Diffusion architecture: GTA MixerUnet
- Trainable parameters: ~29.6M
- Batch size: 256
- Learning rate: `3e-4`
- Training steps: `1,000,000`
- Diffusion sampling steps: 128
- Config: `configs/gta.yaml`

The original diffusion model has been trained successfully.

## Environment

The PyTorch reproduction uses:

- Python 3.9
- PyTorch 2.1
- CUDA
- Gym 0.23
- D4RL
- MuJoCo 2.1
- mujoco-py

## Dataset

The D4RL dataset can be prepared using:

```bash
python data/download_d4rl_datasets.py

For the current experiment:
halfcheetah-medium-v2
observations: (1000000, 17)
actions:      (1000000, 6)
rewards:      (1000000,)

Large generated datasets are intentionally excluded from Git.
Diffusion Training
Foreground:
python src/diffusion/train_diffuser.py \
    --dataset halfcheetah-medium-v2 \
    --config_name gta

Background/server execution:
mkdir -p logs

CUDA_VISIBLE_DEVICES=1 nohup python -u src/diffusion/train_diffuser.py \
    --dataset halfcheetah-medium-v2 \
    --config_name gta \
    > logs/halfcheetah_medium_diffusion.log 2>&1 &

Training progress can be monitored with:
tail -f logs/halfcheetah_medium_diffusion.log

Weights & Biases was also used for experiment tracking.
Checkpoints
The GTA config saves diffusion checkpoints every 200,000 steps.
For the completed HalfCheetah run, checkpoints were generated at:
200000
400000
600000
800000
1000000

Checkpoints, logs, W&B files, and datasets are excluded from Git because of
their size.
Repository Structure
The original GTA implementation is retained as the PyTorch baseline.
Additional work will extend the project with:
- GTA trajectory generation
- TD3+BC downstream offline RL
- JAX/Flax GTA diffusion implementation
- JAX TD3+BC implementation
- PyTorch vs JAX comparisons
Reproduction Notes
Detailed setup and reproduction information is available in:
REPRODUCTION.md
The original upstream README is preserved as:
UPSTREAM_README.md
Attribution
This project builds upon the official GTA implementation by Jaewoo Lee et al.
Original repository:
https://github.com/Jaewoopudding/GTA
The original MIT license is retained in LICENSE.

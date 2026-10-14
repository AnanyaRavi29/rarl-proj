# GTA Reproduction + JAX Port

Reproduction and extension of:

**Generative Trajectory Augmentation with Guidance for Offline Reinforcement Learning (GTA)**
NeurIPS 2024 — original implementation: https://github.com/Jaewoopudding/GTA

This repository contains our reproduction of the original PyTorch GTA implementation, an ongoing
JAX reimplementation, and the **H2 study** (does the best editing strength µ depend on source quality
and local dataset support?).

## Current status

- **PyTorch GTA diffusion model** trained on `halfcheetah-medium-v2` (MixerUnet, ~29.6M parameters,
  batch 256, lr `3e-4`, 1M steps, 128 sampling steps, config `configs/gta.yaml`). Checkpoints at
  200k / 400k / 600k / 800k / 1M steps.
- **GTA editing verified**: edited windows reach +35–75% reward at µ ≥ 0.5 while staying physically
  valid in MuJoCo (`scripts/gta_edit_demo.py`).
- **H2 study** code ready (`src/h2/`, `scripts/h2_run.py`).
- Not done yet: full 5M-transition generation, TD3+BC on GTA data (see *Known issues*).

## 1. Setup (no sudo needed)

Tested on Ubuntu 26.04 with an RTX 5090. Everything installs into your home directory.

```bash
# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash && source ~/.bashrc

# MuJoCo 2.1 binaries (for mujoco-py / D4RL)
mkdir -p ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O ~/mujoco210.tar.gz
tar -xzf ~/mujoco210.tar.gz -C ~/.mujoco

# Environment (compiler + OpenGL headers come from conda-forge instead of apt)
conda create -n gta python=3.9 -y
conda activate gta
conda install -c conda-forge -y gcc=12 gxx=12 mesalib glew glfw patchelf
conda env config vars set \
  LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$CONDA_PREFIX/lib" \
  CPATH="$CONDA_PREFIX/include" LIBRARY_PATH="$CONDA_PREFIX/lib" \
  CC="$CONDA_PREFIX/bin/gcc" CXX="$CONDA_PREFIX/bin/g++"
conda deactivate && conda activate gta

# Python packages. torch 2.1 does not support RTX 50xx GPUs, so torch/triton are installed separately.
grep -vE '^(torch|triton)==' requirements.txt > ~/req_no_torch.txt
pip install -r ~/req_no_torch.txt
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install "cython<3" gdown "faiss-cpu==1.7.4" "numpy==1.21.6"
```

> Keep `numpy==1.21.6`. Newer `faiss-cpu` versions pull in numpy 2, which breaks pandas/scipy here
> (`ValueError: numpy.dtype size changed`).

Check the install (the first `mujoco_py` import compiles itself, ~1 minute):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import mujoco_py, gym, d4rl; print(gym.make('halfcheetah-medium-v2'))"
```

Always `conda activate gta` first — without it `pip` points at the locked system Python and
mujoco-py cannot find `LD_LIBRARY_PATH`.

## 2. Data and checkpoints

Run everything from the repository root.

```bash
# D4RL dataset -> data/halfcheetah-medium-v2.pkl (1000 episodes x 1000 steps)
python data/download_d4rl_datasets.py --env_name=halfcheetah_medium
```

Trained diffusion checkpoints are not in Git (each is ~475 MB). Download them from the team
Google Drive folder (ask a teammate for the link) and place them like this:

```
results/halfcheetah-medium-v2/gta/pretrained/
├── config.yaml
├── model-200000.pt  …  model-1000000.pt
```

```bash
gdown --folder "<drive-folder-url>" -O ~/gdrive_dl
mkdir -p results/halfcheetah-medium-v2/gta/pretrained
mv ~/gdrive_dl/* results/halfcheetah-medium-v2/gta/pretrained/
```

`data/`, `results/`, `outputs/`, `logs/` and checkpoints are gitignored.

## 3. Train the diffusion model (only if you don't have the checkpoints)

```bash
mkdir -p logs
nohup python -u src/diffusion/train_diffuser.py --dataset halfcheetah-medium-v2 --config_name gta \
    > logs/halfcheetah_medium_diffusion.log 2>&1 &
tail -f logs/halfcheetah_medium_diffusion.log
```

Weights & Biases is used for tracking (`wandb login`, or `export WANDB_MODE=offline`).

## 4. How GTA edits data (quick visual demo)

GTA never generates whole 1000-step episodes. It works on **windows** of 31 consecutive steps
(+ the final next-state): one episode gives 1000 − 31 + 1 = 970 overlapping windows, so the dataset
has 970,000. For a window τ⁰ it

1. adds noise **once** at the level picked by the noising ratio µ (µ = 0.5 → σ ≈ 2.4 on normalized data),
2. denoises for µ·128 steps with the trained model, conditioned on 1.3 × the window's return,
3. splits the edited window into 31 transitions (s, a, r, s′) that are added to the RL dataset.

```bash
python scripts/gta_edit_demo.py --ckpt results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt
```

Outputs in `outputs/gta_demo/`: `windows_mu_sweep.png` (velocity and reward, original vs each µ),
`window0_original_vs_edit.gif`, and `metrics.csv`:

| column | meaning |
|---|---|
| `window`, `episode`, `start` | which window (episode index, first step) |
| `mu` | editing strength; `0.0` = the original window |
| `gen_r` | sum of the 31 rewards written by the diffusion model |
| `oracle_r` | sum of the rewards MuJoCo gives for the generated states/actions (true reward) |
| `dyn_mse` | generated vs simulated next state, in dataset-std units (paper Eq. 11) |
| `shift` | how far the edited states moved from the original (dataset-std units) |

## 5. Generate the full GTA dataset (5M transitions)

```bash
mkdir -p data/generated_data/halfcheetah-medium-v2
python src/diffusion/train_diffuser.py --dataset halfcheetah-medium-v2 --config_name gta \
    --load_checkpoint --back_and_forth \
    --ckpt_path results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt
```

Output: `data/generated_data/halfcheetah-medium-v2/gta_smaples.npz` (filename typo is upstream).
If the GPU is shared and you hit CUDA out-of-memory, lower `sample_batch_size` in a copy of the config.

## 6. H2 study: editing strength µ vs. quality and local support

**Question.** GTA uses one µ for the whole dataset. Does the best µ for a window depend on its
**quality Q** (discounted reward inside the window) and its **local support S** (how well the data /
model covers that region)? Hypothesis: best µ rises with S and falls with Q.

**Metrics** (all per 31-step window, converted to percentiles; higher = more support):

| name | definition | training? |
|---|---|---|
| Q | discounted reward inside the window | – |
| S1 | median over steps of the distance to the 20th nearest (s, a) from *other* episodes (FAISS) | no |
| S2 | one-step denoising error of the trained diffusion model at σ ∈ {0.16, 0.5, 2.4}, unguided | no (reuses the GTA model) |
| S3 | exact log-likelihood (probability-flow ODE) on 100 windows, only to validate S2 | no |

**Pipeline** — four stages, each saves its results so later stages and plots can be re-run:

```bash
tmux new -s h2                     # keeps progress bars readable; detach with Ctrl+b then d
conda activate gta
cd rarl-proj

python scripts/h2_run.py pool      # Q, S1, S2, S3 for 20k random windows
python scripts/h2_run.py sweep     # 400 stratified windows x 8 mu x 2 draws, scored in MuJoCo
python scripts/h2_run.py analyze   # plots, statistics, verdict
python scripts/h2_run.py render --window 0   # GIF: original vs edits (camera follows the cheetah)
```

Reattach with `tmux attach -t h2`. The sweep is resumable: rerun the same command after a crash.
Delete `outputs/h2/selected.csv` and `outputs/h2/sweep_parts/` to start the sweep over.

**Expected time** (the GPU is used automatically when ≥ 3 GB is free; force with `--device cpu|cuda`):

| stage | CPU | free GPU |
|---|---|---|
| pool (S1 kNN ~20 min, S2 ~15–25 min, S3 ~5–10 min) | ~40–55 min | ~3–5 min |
| sweep | ~1–2 h | ~10–15 min |
| analyze | ~1 min | ~1 min |

Faster options: `--n_pool 10000`, `--n_likelihood 0` (skip S3), `--draws 1`.
Robustness check with kNN support: `--support S1 --out outputs/h2_s1` for both `sweep` and `analyze`.
All options: `python scripts/h2_run.py --help`.

**Outputs** (`outputs/h2/`):

| file | contents |
|---|---|
| `pool.csv`, `pool_overview.png`, `pool_s3_check.png`, `pool_report.json` | Q/S per window and sanity checks (support spread, correlations) |
| `selected.csv`, `sources.npy` | the 400 stratified windows (5×5 grid on Q × S) |
| `sweep_parts/mu*_d*.npz` | every edited window + its metrics |
| `sweep.csv`, `best_mu.csv` | tidy tables for custom plots |
| `figures/best_mu_map.png` | **main figure**: mean best µ per (quality bin, support bin) |
| `figures/heatmaps.png` | reward gain and dynamic MSE vs µ by quality / support bin |
| `figures/quadrant_curves.png` | gain, dynamic MSE, shift vs µ for the 4 Q × S quadrants |
| `figures/best_mu_vs_Q_S.png` | µ\* and µ₉₀ against Q and S with Spearman correlations |
| `figures/ceiling_check.png` | edited vs source reward (checks the ceiling effect) |
| `summary.md`, `summary.json` | regression with bootstrap CIs, permutation tests, verdict |

**Best µ per window:** `µ*` = highest reward gain with dynamic MSE ≤ 0.017 (2× the paper's average);
`µ₉₀` = smallest µ reaching 90% of the window's maximum gain. **Decision rule (fixed in advance):**
supported if the S coefficient's 95% CI is > 0, the Q coefficient's CI is < 0, and
µ\*(low Q / high S) − µ\*(high Q / low S) ≥ 0.1.

Code layout: `src/h2/common.py` (windows, model loading, MuJoCo checks), `quality.py`, `support.py`,
`edit_eval.py` (µ sweep), `analysis.py` (figures, stats), `render.py`; CLI in `scripts/h2_run.py`.

## 7. Experiment A: does per-window µ help TD3+BC?

Each source window gets its own µ from its support (S2) or quality quintile; all conditions edit the
**same** 161,291 source windows (5M generated transitions) with α = 1.3, then TD3+BC trains on
original + generated data (batch 1024, 1M steps, final score = mean of the last 5 evaluations).

```bash
bash scripts/h2_rl_all.sh          # PyTorch TD3+BC (CORL), 5 conditions x 3 seeds   -> outputs/h2_rl/
bash scripts/h2_rl_jax_all.sh      # JAX TD3+BC (~13x faster), 7 conditions x 3 seeds -> outputs/h2_rl_jax/
bash scripts/h2_quality_pilot.sh   # quick pilot: µ by quality (300k transitions, 300k steps, 2 seeds)
```

JAX environment (training only; generation/evaluation stay in `gta`):
```bash
conda create -n gta-jax python=3.11 -y
~/miniconda3/envs/gta-jax/bin/pip install -U "jax[cuda12]" flax optax numpy tqdm
```

**Results (JAX TD3+BC, D4RL normalised score, mean ± std over 3 seeds)** — `outputs/h2_rl_jax/rl_summary.md`,
bar graph `outputs/h2_rl_jax/figures/rl_results.png`:

| condition | µ by support quintile (low → high) | score |
|---|---|---|
| none | original data only | 48.72 ± 0.06 |
| uniform_0.3 | 0.3 | 51.51 ± 0.36 |
| inverted | 0.4 → 0.2 | 51.81 ± 0.60 |
| adaptive | 0.2 → 0.4 | 54.91 ± 0.41 |
| **uniform_0.5** (paper) | 0.5 | **59.31 ± 0.30** |
| adaptive_wide | 0.3 → 0.7 | 58.60 ± 0.65 |
| inverted_wide | 0.7 → 0.3 | 58.81 ± 0.70 |

PyTorch TD3+BC (`outputs/h2_rl/`) gives the same picture (none 48.65, uniform_0.5 58.67, adaptive 53.53).

**Quality pilot** (`outputs/h2_quality_pilot/`, µ by quality quintile ranked within support quintiles, mean µ 0.5):
uniform 56.10 ± 0.16, quality_inverted 55.59 ± 0.30, quality_adaptive (edit good windows less) 54.86 ± 0.23.

**Conclusions so far**
- Reproduction matches the paper: no augmentation ≈ 48.7, GTA µ = 0.5 ≈ 58.7–59.3 (paper 48.4 / 57.8).
- With a weak editing budget (mean µ 0.3), giving more editing to well-supported windows helps (+3 points, p < 0.01, PyTorch and JAX).
- At GTA's strength (mean µ 0.5), assigning µ by support or by quality does **not** beat a single µ = 0.5.
- Support still predicts which edits break physics (H2 study, section 6); it just does not translate into RL gains at µ = 0.5.
- Caveats: one dataset, 3 training seeds, one generated dataset per condition, support and quality correlated (ρ ≈ 0.42), guidance w = 0.

Slides on the metrics (quality, S1, S2, S3, validation): `docs/h2_metrics/h2_metrics.pdf`; full deck: `docs/h2_presentation/`.

## Known issues

- **TD3+BC with GTA data (`corl/algorithms/td3_bc.py --GDA GTA`) crashes as is**:
  the loader reads `data/generated_data/<env>.npz` but generation writes
  `data/generated_data/<env>/gta_smaples.npz`; `corl/shared/utils.py` reads the key
  `guidance_rewardscale` (config calls it `amplify_returnscale`); `get_dataset` uses `config.step`,
  which `TrainConfig` lacks.
- **Guidance strength:** `temperature: 1.0` in the config means w = 0 (pure conditional sampling);
  the paper reports w = 2.0 (`temperature: 3.0`).
- The denoising network drops the return condition at random (p = 0.25) even during sampling (upstream behavior).
- Use TD3+BC batch size 1024 to match the paper (`--batch_size 1024`; CORL default is 256).

## More

- Reproduction notes: `REPRODUCTION.md`
- Original upstream README: `UPSTREAM_README.md`

## Attribution

This project builds upon the official GTA implementation by Jaewoo Lee et al.
(https://github.com/Jaewoopudding/GTA). The original MIT license is retained in `LICENSE`.

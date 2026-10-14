#!/usr/bin/env bash
# Experiment A with JAX TD3+BC: 7 conditions x 3 seeds, then the same report/bar graph.
# Run from the repo root inside tmux:  bash scripts/h2_rl_jax_all.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
OUT=outputs/h2_rl_jax
SEEDS=${SEEDS:-"0 1 2"}
PAR=${PAR:-7}                      # JAX runs in parallel on the GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT/logs" && ln -sfn ../h2_rl/data "$OUT/data"   # data-quality table in the report

train() {  # condition seed
  local run="h2rl_$1_seed$2"
  if [ -f "jax_impl/checkpoints/$run/actor_step_1000000.npz" ]; then return; fi
  conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py \
    --dataset "$OUT/jax_data/$1.npz" --run-name "$run" --seed "$2" \
    --batch-size 1024 --steps 1000000 --checkpoint-every 10000 > "$OUT/logs/train_$1_seed$2.log" 2>&1
}

# 1) the two wide GTA datasets (PyTorch diffusion) in the background
conda activate gta
python scripts/h2_rl.py gen --out outputs/h2_rl --conditions adaptive_wide inverted_wide > "$OUT/logs/gen_wide.log" 2>&1 &
GEN=$!

# 2) JAX data + training for the 5 conditions whose GTA data already exist
python scripts/h2_rl_jax.py data --conditions none uniform_0.3 adaptive inverted uniform_0.5
for seed in $SEEDS; do for cond in none uniform_0.3 adaptive inverted uniform_0.5; do
  while [ "$(jobs -rp | wc -l)" -gt "$PAR" ]; do wait -n || true; done   # +1 for the generator
  train "$cond" "$seed" &
done; done

# 3) wide conditions once their data exist
wait $GEN
python scripts/h2_rl_jax.py data --conditions adaptive_wide inverted_wide
for seed in $SEEDS; do for cond in adaptive_wide inverted_wide; do train "$cond" "$seed" & done; done
wait

# 4) evaluate every run in MuJoCo, then the report and bar graph
python scripts/h2_rl_jax.py eval --out "$OUT"
python scripts/h2_rl.py report --out "$OUT"
echo "Done: $OUT/rl_summary.md, $OUT/figures/rl_results.png"

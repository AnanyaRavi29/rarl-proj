#!/usr/bin/env bash
# Alpha experiment end to end (run in tmux from the repo root): bash scripts/h2_alpha.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h2_alpha
mkdir -p $OUT/logs

train_all() {  # condition names...
  for c in "$@"; do for s in 0 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n || true; done
    [ -f jax_impl/checkpoints/alpha__${c}_seed$s/actor_step_1000000.npz ] && continue
    conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py --dataset $OUT/jax_data/$c.npz \
      --run-name alpha__${c}_seed$s --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 \
      > $OUT/logs/train_${c}_seed$s.log 2>&1 &
  done; done
  wait
}

conda activate gta
echo "== stage 1: global alpha"; python scripts/h2_alpha.py gen --stage 1
train_all alpha_1 alpha_1.1 alpha_1.3 alpha_1.5 alpha_1.8 alpha_2
python scripts/h2_alpha.py eval && python scripts/h2_alpha.py report

echo "== stage 2: alpha by support around the best alpha"; python scripts/h2_alpha.py gen --stage 2
train_all alpha_up alpha_down
python scripts/h2_alpha.py eval && python scripts/h2_alpha.py report
echo "Done: $OUT/summary.md, $OUT/alpha_bars.png"

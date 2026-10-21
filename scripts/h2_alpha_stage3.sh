#!/usr/bin/env bash
# Stage 3: alpha_random + fresh seeds 10-14 for uniform/down/random/up. Waits for a free GPU before GPU work.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h2_alpha
NEED_MB=${NEED_MB:-8000}
mkdir -p $OUT/logs

wait_gpu() {  # block until the GPU has NEED_MB free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$1" ] && return
    echo "$(date +%H:%M) waiting for GPU: ${free} MiB free, need $1"; sleep 60
  done
}

conda activate gta
wait_gpu $NEED_MB
python scripts/h2_alpha.py gen --stage 3

for c in alpha_1.3 alpha_down alpha_random alpha_up; do for s in 10 11 12 13 14; do
  while [ "$(jobs -rp | wc -l)" -ge 5 ]; do wait -n || true; done
  [ -f jax_impl/checkpoints/alpha__${c}_seed$s/actor_step_1000000.npz ] && continue
  wait_gpu 3000
  conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py --dataset $OUT/jax_data/$c.npz \
    --run-name alpha__${c}_seed$s --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 \
    > $OUT/logs/train_${c}_seed$s.log 2>&1 &
  sleep 20
done; done
wait
python scripts/h2_alpha.py eval && python scripts/h2_alpha.py report
echo "Done: $OUT/summary.md, $OUT/alpha_stage3_bars.png"

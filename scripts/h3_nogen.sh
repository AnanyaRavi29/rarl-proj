#!/usr/bin/env bash
# H3 end to end in tmux: bash scripts/h3_nogen.sh   (results in outputs/h3_nogen/)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h3_nogen
mkdir -p $OUT/logs
conda activate gta
python scripts/h3_nogen.py data

wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 3000 ]; do
  echo "$(date +%H:%M) waiting for GPU"; sleep 60; done; }

for c in none rw_b0.5 rw_b1 rw_b2 top25 top10 gta_ref; do for s in 0 1 2; do
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n || true; done
  [ -f jax_impl/checkpoints/h3__${c}_seed$s/actor_step_1000000.npz ] && continue
  wait_gpu
  ~/miniconda3/envs/gta-jax/bin/python jax_impl/train_td3bc.py --dataset $OUT/jax_data/$c.npz --run-name h3__${c}_seed$s \
    --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 > $OUT/logs/train_${c}_seed$s.log 2>&1 &
  sleep 15
done; done
wait
python scripts/h3_nogen.py eval && python scripts/h3_nogen.py report
echo "Done: $OUT/summary.md, $OUT/h3_bars.png"

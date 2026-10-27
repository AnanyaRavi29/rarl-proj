#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h6_twga/gate0
mkdir -p $OUT/logs
conda activate gta
python scripts/h6_gate0.py errors
python scripts/h6_gate0.py data
for a in 1.5 1.8 2; do for s in 0 1 2; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 3000 ]; do sleep 60; done
  ~/miniconda3/envs/gta-jax/bin/python jax_impl/train_td3bc_w.py --dataset $OUT/jax_data/w_alpha_$a.npz \
    --run-name h6g0__w_alpha_${a}_seed$s --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 \
    > $OUT/logs/train_${a}_seed$s.log 2>&1 &
  sleep 10
done; done
wait
python scripts/h6_gate0.py eval && python scripts/h6_gate0.py report
echo "Done: $OUT/summary.md"

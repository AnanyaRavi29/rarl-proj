#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p outputs/h2_quality_pilot
conda activate gta
python scripts/h2_quality_pilot.py gen
for c in uniform_0.5 quality_adaptive quality_inverted; do for s in 0 1; do
  conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py --dataset outputs/h2_quality_pilot/jax_data/$c.npz \
    --run-name qpilot_${c}_seed$s --seed $s --batch-size 1024 --steps 300000 --checkpoint-every 10000 \
    > outputs/h2_quality_pilot/train_${c}_seed$s.log 2>&1 &
done; done
wait
python scripts/h2_quality_pilot.py eval

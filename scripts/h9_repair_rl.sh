#!/usr/bin/env bash
# Repaired-GTA TD3+BC runs, 3 alphas x 3 seeds in parallel (run from the repo root): bash scripts/h9_repair_rl.sh
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h9_repair
mkdir -p $OUT/logs
for a in 1.5 1.8 2; do for s in 0 1 2; do
  [ -f jax_impl/checkpoints/repair__repair_alpha_${a}_seed$s/actor_step_1000000.npz ] && continue
  conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py --dataset $OUT/jax_data/repair_alpha_$a.npz \
    --run-name repair__repair_alpha_${a}_seed$s --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 \
    > $OUT/logs/train_alpha_${a}_seed$s.log 2>&1 &
done; done
wait
echo "all runs done"

#!/usr/bin/env bash
# Acceleration-edit experiment, chained (run in tmux from the repo root):
#   tmux new -d -s accel "bash scripts/h12_pipeline.sh"
# Progress: outputs/h12_accel/pipeline.log ; results: outputs/h12_accel/summary.md
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h12_accel
K=6
ALPHAS="1.5 1.8"
mkdir -p $OUT/logs
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/pipeline.log; }

log "== waiting for the whole-window repair pipeline to free the GPU"
until grep -q "pipeline finished" outputs/h10_window_repair/pipeline.log 2>/dev/null; do sleep 30; done

log "== step 1: generate acceleration edits (k=$K pinned steps)"
for a in $ALPHAS; do
  conda run --no-capture-output -n gta python scripts/h12_accel.py gen --alpha $a --k $K >> $OUT/logs/gen_alpha_$a.log 2>&1 \
    && log "data alpha $a done: $(tr -d '\n ' < $OUT/check_accel_k${K}_alpha_$a.json)" \
    || log "data alpha $a FAILED (see logs/gen_alpha_$a.log)"
done

for s in 0 1 2; do
  log "== step 2.$s: TD3+BC seed $s"
  for a in $ALPHAS; do
    name=accel_k${K}_alpha_$a
    [ -f $OUT/jax_data/$name.npz ] || continue
    [ -f jax_impl/checkpoints/accel__${name}_seed$s/actor_step_1000000.npz ] && continue
    conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py --dataset $OUT/jax_data/$name.npz \
      --run-name accel__${name}_seed$s --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 \
      > $OUT/logs/train_${name}_seed$s.log 2>&1 &
    sleep 20
  done
  wait
  log "seed $s training finished"
  conda run --no-capture-output -n gta python scripts/h12_accel.py eval >> $OUT/logs/eval.log 2>&1 \
    && log "evaluation: $(tail -n +5 $OUT/summary.md | head -2 | tr '\n' ' ')" || log "evaluation FAILED"
done
log "== pipeline finished; summary: $OUT/summary.md"

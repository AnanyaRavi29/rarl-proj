#!/usr/bin/env bash
# Whole-window (collocation) repair, end to end, one step after the other. Run inside tmux from the repo root:
#   tmux new -d -s wrepair "bash scripts/h10_pipeline.sh"
# Progress: outputs/h10_window_repair/pipeline.log ; results: outputs/h10_window_repair/summary_*.md
# A failed training run does not stop the pipeline; finished runs are skipped on a restart.
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=outputs/h10_window_repair
ALPHAS="1.5 1.8 2"
mkdir -p $OUT/logs
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/pipeline.log; }

log "== step 1: build repaired datasets"
for a in $ALPHAS; do
  if [ -f $OUT/jax_data/wrepair_alpha_$a.npz ]; then log "dataset alpha $a exists, skipping"; continue; fi
  conda run --no-capture-output -n gta python scripts/h10_window_repair.py data --mode colloc --alphas $a \
    --lams 0.1 --iters 500 --batch 4096 >> $OUT/logs/data_alpha_$a.log 2>&1 \
    && log "dataset alpha $a done" || log "dataset alpha $a FAILED (see logs/data_alpha_$a.log)"
done

train_seed() {  # one seed for every alpha, 3 runs in parallel (GPU memory limit)
  local s=$1
  for a in $ALPHAS; do
    [ -f $OUT/jax_data/wrepair_alpha_$a.npz ] || continue
    [ -f jax_impl/checkpoints/wrepair__wrepair_alpha_${a}_seed$s/actor_step_1000000.npz ] && continue
    conda run -n gta-jax --no-capture-output python jax_impl/train_td3bc.py \
      --dataset $OUT/jax_data/wrepair_alpha_$a.npz --run-name wrepair__wrepair_alpha_${a}_seed$s --seed $s \
      --batch-size 1024 --steps 1000000 --checkpoint-every 10000 > $OUT/logs/train_alpha_${a}_seed$s.log 2>&1 &
    sleep 20  # stagger start-up so memory peaks do not coincide
  done
  wait
}

evaluate() {
  conda run --no-capture-output -n gta python scripts/h10_window_repair.py eval --upto 1000000 \
    >> $OUT/logs/eval.log 2>&1 && log "evaluation done: $(tail -n +3 $OUT/summary_1000000.md | head -5 | tr '\n' ' ')" \
    || log "evaluation FAILED (see logs/eval.log)"
}

for s in 0 1 2; do
  log "== step 2.$s: TD3+BC seed $s (alphas $ALPHAS)"
  train_seed $s
  log "seed $s training finished"
  evaluate
done
log "== pipeline finished; summary: $OUT/summary_1000000.md"

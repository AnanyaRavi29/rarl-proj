#!/usr/bin/env bash
# Experiment A end to end: generate the 4 GTA datasets, then TD3+BC for 5 conditions x SEEDS seeds
# (one seed at a time, the 5 conditions in parallel on the GPU), then the report.
# Run inside tmux from the repo root:  bash scripts/h2_rl_all.sh
set -e
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-outputs/h2_rl}
mkdir -p "$OUT/logs"

python scripts/h2_rl.py gen --out "$OUT" 2>&1 | tee "$OUT/logs/gen.log"

for seed in $SEEDS; do
  for cond in none uniform_0.5 uniform_0.3 adaptive inverted; do
    python scripts/h2_rl.py train --condition "$cond" --seed "$seed" --out "$OUT" \
      > "$OUT/logs/${cond}_seed${seed}.log" 2>&1 &
  done
  echo "seed $seed: 5 TD3+BC runs started ($(date)). Progress: tail -f $OUT/logs/*_seed${seed}.log"
  wait
  python scripts/h2_rl.py report --out "$OUT" > "$OUT/logs/report.log" 2>&1 || true
done

python scripts/h2_rl.py report --out "$OUT"

#!/usr/bin/env bash
# Experiment A2: support-adaptive mu centred on 0.5 (no cap). Reuses sources + uniform_0.5 runs.
set -e
OUT=${OUT:-outputs/h2_rl}
python scripts/h2_rl.py gen --out "$OUT" --conditions adaptive_wide inverted_wide 2>&1 | tee "$OUT/logs/gen_wide.log"
for cond in adaptive_wide inverted_wide; do
  for seed in 0 1 2; do
    python scripts/h2_rl.py train --condition "$cond" --seed "$seed" --out "$OUT" > "$OUT/logs/${cond}_seed${seed}.log" 2>&1 &
  done
done
wait
python scripts/h2_rl.py report --out "$OUT"

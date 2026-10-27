#!/usr/bin/env bash
# Generate with the VAE and train JAX TD3+BC (3 seeds); results in outputs/h4_vae/
set -e
source ~/miniconda3/etc/profile.d/conda.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
A=${A:-1.3}; L=${L:-1.0}; NAME=vae_a${A}_l${L}
conda activate gta
python scripts/h4_vae.py gen --alpha $A --lam $L
mkdir -p outputs/h4_vae/logs
for s in 0 1 2; do
  ~/miniconda3/envs/gta-jax/bin/python jax_impl/train_td3bc.py --dataset outputs/h4_vae/jax_data/${NAME}.npz \
    --run-name h4__${NAME}_seed$s --seed $s --batch-size 1024 --steps 1000000 --checkpoint-every 10000 \
    > outputs/h4_vae/logs/train_${NAME}_seed$s.log 2>&1 &
done
wait
OMP_NUM_THREADS=1 python -c "
import numpy as np, json, pyrootutils
pyrootutils.setup_root('scripts/h4_vae.py', indicator='.aug-project-root', pythonpath=True, cwd=True)
from pathlib import Path
from scripts.h2_rl_jax import _evaluate
res = {}
for s in [0, 1, 2]:
    r = f'h4__${NAME}_seed{s}'
    rows = _evaluate(Path('jax_impl/checkpoints') / r, [960000, 970000, 980000, 990000, 1000000], 10, 0)
    res[r] = float(np.mean([v for _, v in rows])); print(r, round(res[r], 2), flush=True)
v = list(res.values()); print(f'${NAME}: {np.mean(v):.2f} +- {np.std(v, ddof=1):.2f}')
f = Path('outputs/h4_vae/results.json'); old = json.loads(f.read_text()) if f.exists() else {}; old.update(res); f.write_text(json.dumps(old, indent=1))
"

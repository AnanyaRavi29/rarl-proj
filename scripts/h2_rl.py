"""Experiment A: does giving each window a support-dependent mu improve TD3+BC?

    python scripts/h2_rl.py gen                                # support + 4 GTA datasets (GPU)
    python scripts/h2_rl.py train --condition adaptive --seed 0
    python scripts/h2_rl.py report                             # curves, scores, verdict
Full pipeline in tmux: bash scripts/h2_rl_all.sh
"""
import argparse
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

DEFAULT_CKPT = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"


def cmd_gen(args):
    import gym
    import d4rl  # noqa: F401
    from src.h2.common import WindowData, build_model, load_cfg, pick_device
    from src.h2.rl_data import CONDITIONS, generate_condition, select_sources

    cfg = load_cfg(args.ckpt, args.config)
    wd = WindowData(args.env, cfg.Dataset.seq_len, cfg.Dataset.gamma, cfg.Dataset.get("penalty", 100))
    assert not any(np.asarray(e["rew"]).min() < -50 for e in wd.eps), "datasets with terminals are not handled here"
    device = pick_device(args.device)
    model = build_model(args.ckpt, cfg, wd, device)
    out = Path(args.out) / "data"
    out.mkdir(parents=True, exist_ok=True)
    src = select_sources(wd, model, cfg.Dataset.gamma, args.n_windows, [0.16, 0.5, 2.4], 4, args.batch,
                         device, args.seed, out)
    env = gym.make(args.env)
    for name in (args.conditions or list(CONDITIONS)):
        generate_condition(name, src, wd, model, env, cfg, args.batch, device, args.seed, out)


def cmd_train(args):
    from src.h2.rl_train import run_td3bc
    gta = None if args.condition == "none" else Path(args.out) / "data" / f"{args.condition}.npz"
    if gta is not None and not gta.exists():
        raise SystemExit(f"{gta} not found - run the gen stage first")
    run_td3bc(args.env, args.condition, gta, args.seed, Path(args.out) / "runs", steps=args.steps,
              batch_size=1024, eval_freq=args.eval_freq, device=args.device if args.device != "auto" else "cuda")


def cmd_report(args):
    from src.h2.rl_report import report
    report(args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "train", "report"])
    p.add_argument("--env", default="halfcheetah-medium-v2")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="outputs/h2_rl")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    # gen
    p.add_argument("--n_windows", type=int, default=161_291, help="5M transitions / 31, as upstream GTA")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--conditions", nargs="+", default=None)
    # train
    p.add_argument("--condition", default="adaptive")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval_freq", type=int, default=10_000)
    args = p.parse_args()
    {"gen": cmd_gen, "train": cmd_train, "report": cmd_report}[args.stage](args)


if __name__ == "__main__":
    main()

"""H2 study: does the best GTA editing strength mu depend on source quality Q and local support S?

Stages (run from the repo root, `gta` conda env):
    python scripts/h2_run.py pool       # Q, S1, S2 (+ S3 check) for a random pool of windows
    python scripts/h2_run.py sweep      # stratified sample -> edit at every mu -> MuJoCo metrics (resumable)
    python scripts/h2_run.py analyze    # trend plots, statistics, verdict -> outputs/h2/summary.md
    python scripts/h2_run.py render --window 0   # GIF: original vs edits for one window
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

DEFAULT_CKPT = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"
DEFAULT_MUS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75]


def setup(args, need_model=True):
    import gym
    import d4rl  # noqa: F401
    from src.h2.common import WindowData, build_model, load_cfg, pick_device

    cfg = load_cfg(args.ckpt, args.config)
    wd = WindowData(args.env, cfg.Dataset.seq_len, cfg.Dataset.gamma, cfg.Dataset.get("penalty", 100))
    device = pick_device(args.device)
    model = build_model(args.ckpt, cfg, wd, device) if need_model else None
    return cfg, wd, model, gym.make(args.env), device


def cmd_pool(args):
    from src.h2.analysis import plot_pool, pool_report
    from src.h2.quality import build_catalog
    from src.h2.support import knn_distance, log_likelihood, recon_error, recon_score

    cfg, wd, model, _, device = setup(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    catalog = build_catalog(wd, cfg.Dataset.gamma, args.max_start)
    pool = catalog.sample(min(args.n_pool, len(catalog)), random_state=args.seed).reset_index(drop=True)
    print(f"Catalog: {len(catalog)} windows -> pool of {len(pool)}")
    es, ss = pool.episode.values, pool.start.values

    pool["s1_dist"] = knn_distance(wd, es, ss, k=args.k)
    errs = recon_error(model, wd, es, ss, args.sigmas, args.recon_draws, args.batch, device, args.seed)
    for j, sig in enumerate(args.sigmas):
        pool[f"s2_err_sigma{sig:g}"] = errs[:, j]
    pool["s2_score"] = recon_score(errs)
    pool["s3_loglik"] = np.nan
    if args.n_likelihood > 0:
        idx = np.random.default_rng(args.seed).choice(len(pool), min(args.n_likelihood, len(pool)), replace=False)
        pool.loc[idx, "s3_loglik"] = log_likelihood(model, wd, es[idx], ss[idx], device=device)

    pool["Q"] = pool.q_return.rank(pct=True)
    pool["S1"] = (-pool.s1_dist).rank(pct=True)
    pool["S2"] = pool.s2_score.rank(pct=True)
    pool.to_csv(out / "pool.csv", index=False)
    plot_pool(pool, out)
    rep = pool_report(pool)
    (out / "pool_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"Saved {out / 'pool.csv'}, {out / 'pool_overview.png'}")


def cmd_sweep(args):
    import pandas as pd
    from src.h2.common import edit_metrics
    from src.h2.edit_eval import amplified_condition, run_sweep, stratified_sample

    cfg, wd, model, env, device = setup(args)
    out = Path(args.out)
    sel_path = out / "selected.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        print(f"Reusing {sel_path} ({len(sel)} windows)")
    else:
        pool = pd.read_csv(out / "pool.csv")
        sel = stratified_sample(pool, args.support, args.bins, args.per_cell, args.seed)
    sources = wd.windows(sel.episode.values, sel.start.values)
    state_std = wd.state_std()
    if not sel_path.exists():
        src_m = edit_metrics(env, sources, sources, wd.obs_dim, wd.act_dim, state_std, desc="oracle sources")
        sel["src_oracle_r"] = src_m["oracle_r"]
        sel.to_csv(sel_path, index=False)
        np.save(out / "sources.npy", sources)
        print(f"Selected {len(sel)} windows ({args.bins}x{args.bins} grid on Q x {args.support})")

    g = cfg.SimpleDiffusionGenerator
    cond = amplified_condition(sel.rtg.values, g.amplify_returnscale, cfg.Dataset.reward_scale)
    run_sweep(model, env, sources, cond, args.mus, args.draws, args.batch, g.temperature, g.num_sample_steps,
              wd.obs_dim, wd.act_dim, state_std, out, device, args.seed)
    print(f"Done. Next: python scripts/h2_run.py analyze --out {out}")


def cmd_analyze(args):
    from src.h2.analysis import analyze
    analyze(Path(args.out), args.support, args.thr, args.n_boot, args.seed)


def cmd_render(args):
    from src.h2.render import render_window
    _, wd, _, env, _ = setup(args, need_model=False)
    render_window(env, args.out, args.window, args.mus, args.draw, wd.obs_dim, wd.act_dim)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["pool", "sweep", "analyze", "render"])
    p.add_argument("--env", default="halfcheetah-medium-v2")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--config", default=None, help="defaults to config.yaml next to the checkpoint")
    p.add_argument("--out", default="outputs/h2")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=200, help="windows per model call")
    # pool
    p.add_argument("--n_pool", type=int, default=20000)
    p.add_argument("--max_start", type=int, default=800)
    p.add_argument("--k", type=int, default=20, help="S1 neighbour rank")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.16, 0.5, 2.4], help="S2 noise levels")
    p.add_argument("--recon_draws", type=int, default=4)
    p.add_argument("--n_likelihood", type=int, default=100, help="S3 subset size (0 = skip)")
    # sweep
    p.add_argument("--support", default="S2", choices=["S1", "S2"])
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--per_cell", type=int, default=16)
    p.add_argument("--mus", type=float, nargs="+", default=DEFAULT_MUS)
    p.add_argument("--draws", type=int, default=2)
    # analyze
    p.add_argument("--thr", type=float, default=0.017, help="dynamic-MSE limit for mu*")
    p.add_argument("--n_boot", type=int, default=2000)
    # render
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--draw", type=int, default=0)
    args = p.parse_args()
    if args.stage == "render" and args.mus == DEFAULT_MUS:
        args.mus = [0.2, 0.35, 0.5, 0.75]
    {"pool": cmd_pool, "sweep": cmd_sweep, "analyze": cmd_analyze, "render": cmd_render}[args.stage](args)


if __name__ == "__main__":
    main()

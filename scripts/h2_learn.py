"""Learned continuous mu and alpha (CMA-ES over a smooth map of support S and quality Q).

Phases (run in the `gta` env; TD3+BC runs as JAX subprocesses in `gta-jax`):
  global   CMA-ES over [a0, c0] only: the best single mu and alpha       (baseline, same optimiser)
  learned  CMA-ES over [a0, aS, aQ, c0, cS, cQ]: mu and alpha vary with S and Q
  confirm  best global vs best learned vs GTA default (mu 0.5, alpha 1.3), 1M transitions, 5 fresh seeds

Search candidates use 16,129 windows (500k generated transitions) and 2 TD3+BC seeds (0, 1), with the
same windows, generation seed and RL seeds for every candidate (common random numbers).
Everything is cached in outputs/h2_learn/history.jsonl, so a restart resumes where it stopped.

    python scripts/h2_learn.py run          # all phases
    python scripts/h2_learn.py report       # figures + summary from what exists
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h2_learn")
ENV = "halfcheetah-medium-v2"
CKPT = Path("jax_impl/checkpoints")
JAX_PY = str(Path.home() / "miniconda3/envs/gta-jax/bin/python")
SEARCH_WINDOWS, CONFIRM_WINDOWS = 16_129, 32_258
SEARCH_SEEDS, CONFIRM_SEEDS = [0, 1], [20, 21, 22, 23, 24]
LOWER = np.array([-3, -8, -8, -3, -8, -8], float)
UPPER = -LOWER


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def wait_gpu(need_mb):
    while gpu_free_mb() < need_mb:
        log(f"waiting for GPU ({gpu_free_mb()} MiB free, need {need_mb})")
        time.sleep(60)


def key_of(phase, x):
    return phase + "_" + hashlib.md5(np.round(x, 4).tobytes()).hexdigest()[:10]


def load_history():
    f = OUT / "history.jsonl"
    return {json.loads(l)["key"]: json.loads(l) for l in f.read_text().splitlines()} if f.exists() else {}


class Runner:
    def __init__(self):
        import gym
        import d4rl  # noqa: F401
        from src.h2.common import WindowData, build_model, load_cfg
        ckpt = "results/halfcheetah-medium-v2/gta/pretrained/model-1000000.pt"
        self.cfg = load_cfg(ckpt)
        self.wd = WindowData(ENV, self.cfg.Dataset.seq_len, self.cfg.Dataset.gamma)
        wait_gpu(8000)
        self.model = build_model(ckpt, self.cfg, self.wd, "cuda")
        self.env = gym.make(ENV)
        self.state_std = self.wd.state_std()
        full = dict(np.load("outputs/h2_rl/data/sources.npz"))
        self.windows = {}
        for name, n, seed in [("search", SEARCH_WINDOWS, 2), ("confirm", CONFIRM_WINDOWS, 5)]:
            ix = np.sort(np.random.default_rng(seed).choice(len(full["episode"]), n, replace=False))
            src = {k: v[ix] for k, v in full.items()}
            src["S"] = (np.argsort(np.argsort(src["s2_score"])) + 0.5) / n - 0.5   # S2 support percentile, centred
            src["Q"] = (np.argsort(np.argsort(src["q_return"])) + 0.5) / n - 0.5   # quality percentile, centred
            self.windows[name] = src
        self.history = load_history()

    def generate(self, x, which, path):
        """Edit every window with its own mu and alpha; write the flat JAX training file."""
        import torch
        from src.h2.edit_eval import amplified_condition
        from src.h2.learned import mu_alpha, sample_per_window
        from src.h2.common import edit_metrics
        from src.h2.rl_data import windows_to_transitions
        from src.h2.rl_train import load_dataset
        src = self.windows[which]
        mus, alphas = mu_alpha(x, src["S"], src["Q"])
        cond = amplified_condition(src["rtg"], alphas, self.cfg.Dataset.reward_scale)
        n, D = len(mus), self.wd.obs_dim + self.wd.act_dim + 1
        edited = np.empty((n, self.wd.H + 1, D), np.float32)
        torch.manual_seed(0)
        wait_gpu(8000)
        for b in range(0, n, 2048):
            j = slice(b, b + 2048)
            w = torch.from_numpy(self.wd.windows(src["episode"][j], src["start"][j]))
            c = torch.from_numpy(cond[j]).to("cuda")
            edited[j] = sample_per_window(self.model, w, c, mus[j],
                                          self.cfg.SimpleDiffusionGenerator.temperature).cpu().numpy()
        chk = np.random.default_rng(0).choice(n, 300, replace=False)
        srcw = self.wd.windows(src["episode"][chk], src["start"][chk])
        m = edit_metrics(self.env, edited[chk], srcw, self.wd.obs_dim, self.wd.act_dim, self.state_std, desc="check")
        ms = edit_metrics(self.env, srcw, srcw, self.wd.obs_dim, self.wd.act_dim, self.state_std, desc="src")
        gen = OUT / "tmp" / (path.stem + "_gen.npz")
        np.savez(gen, **windows_to_transitions(edited, self.wd.obs_dim, self.wd.act_dim))
        np.savez(path, **load_dataset(ENV, gen))
        gen.unlink()
        return dict(mean_mu=float(mus.mean()), sd_mu=float(mus.std()), mean_alpha=float(alphas.mean()),
                    sd_alpha=float(alphas.std()), oracle_r=float(m["oracle_r"].mean()),
                    src_r=float(ms["oracle_r"].mean()), dyn_mse=float(m["dyn_mse"].mean()),
                    shift=float(m["shift"].mean()))

    def train(self, data_path, run, seed):
        wait_gpu(3000)
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        logf = open(OUT / "logs" / f"{run}.log", "w")
        return subprocess.Popen([JAX_PY, "jax_impl/train_td3bc.py", "--dataset", str(data_path), "--run-name", run,
                                 "--seed", str(seed), "--batch-size", "1024", "--steps", "1000000",
                                 "--checkpoint-every", "10000"], stdout=logf, stderr=subprocess.STDOUT, env=env)

    def evaluate_batch(self, phase, xs, which, seeds):
        """Generate + train + evaluate a list of parameter vectors; returns mean scores (cached)."""
        pending, procs = [], []
        for x in xs:
            k = key_of(phase, x)
            if k in self.history:
                continue
            data_path = OUT / "tmp" / f"{k}.npz"
            log(f"[{phase}] generating {k}: x = {np.round(x, 3).tolist()}")
            info = self.generate(x, which, data_path)
            log(f"[{phase}] {k}: mean mu {info['mean_mu']:.3f} (sd {info['sd_mu']:.3f}), "
                f"mean alpha {info['mean_alpha']:.3f} (sd {info['sd_alpha']:.3f}), "
                f"edited reward {info['oracle_r']:.1f} vs {info['src_r']:.1f}, dyn {info['dyn_mse']:.4f}")
            runs = [f"learn__{k}_seed{s}" for s in seeds]
            procs += [self.train(data_path, r, s) for r, s in zip(runs, seeds)]
            pending.append((k, x, info, runs, data_path))
        for p in procs:
            p.wait()
        from scripts.h2_rl_jax import _evaluate
        with ProcessPoolExecutor(8) as ex:
            for k, x, info, runs, data_path in pending:
                futs = [ex.submit(_eval_last5, r) for r in runs]
                scores = [f.result() for f in futs]
                rec = dict(key=k, phase=phase, x=np.round(x, 5).tolist(), scores=scores,
                           score=float(np.mean(scores)), **info)
                self.history[k] = rec
                with open(OUT / "history.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                data_path.unlink(missing_ok=True)
                for r in runs:  # keep only the final actor
                    for fp in (CKPT / r).glob("*"):
                        if not fp.name.startswith(("actor_step_1000000", "normalization", "config")):
                            fp.unlink()
                log(f"[{phase}] {k}: scores {np.round(scores, 2).tolist()} -> {rec['score']:.2f}")
        return [self.history[key_of(phase, x)]["score"] for x in xs]


def _eval_last5(run):
    os.environ["OMP_NUM_THREADS"] = "1"
    from scripts.h2_rl_jax import _evaluate
    rows = _evaluate(CKPT / run, [960_000, 970_000, 980_000, 990_000, 1_000_000], 10, 0)
    return float(np.mean([v for _, v in rows]))


def cma_phase(runner, phase, dims, popsize, generations, sigma0):
    import cma
    from src.h2.learned import default_params
    x_full0 = default_params()
    def full(z):
        x = x_full0.copy()
        x[dims] = z
        return np.clip(x, LOWER, UPPER)
    es = cma.CMAEvolutionStrategy(x_full0[dims], 1.0, {
        "popsize": popsize, "seed": 1, "CMA_stds": sigma0, "bounds": [LOWER[dims].tolist(), UPPER[dims].tolist()],
        "verbose": -9})
    runner.evaluate_batch(phase, [x_full0], "search", SEARCH_SEEDS)  # the GTA default, for reference
    for g in range(generations):
        zs = es.ask()
        scores = runner.evaluate_batch(phase, [full(z) for z in zs], "search", SEARCH_SEEDS)
        es.tell(zs, [-s for s in scores])
        best = max((r for r in runner.history.values() if r["phase"] == phase), key=lambda r: r["score"])
        log(f"[{phase}] generation {g + 1}/{generations} done; best so far {best['score']:.2f} at "
            f"mean mu {best['mean_mu']:.3f}, mean alpha {best['mean_alpha']:.3f}")


def best_of(history, phase):
    return max((r for r in history.values() if r["phase"] == phase), key=lambda r: r["score"])


def cmd_run(args):
    for d in ["tmp", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    runner = Runner()
    # intercepts move ~0.7 logit units, slopes ~3 (S', Q' span 1)
    cma_phase(runner, "global", [0, 3], popsize=4, generations=args.global_gens, sigma0=[0.7, 0.7])
    cma_phase(runner, "learned", [0, 1, 2, 3, 4, 5], popsize=6, generations=args.learned_gens,
              sigma0=[0.7, 3, 3, 0.7, 3, 3])
    from src.h2.learned import default_params
    finals = {"default": default_params(), "best_global": np.array(best_of(runner.history, "global")["x"]),
              "best_learned": np.array(best_of(runner.history, "learned")["x"])}
    json.dump({k: v.tolist() for k, v in finals.items()}, open(OUT / "finalists.json", "w"), indent=1)
    for name, x in finals.items():
        runner.evaluate_batch(f"confirm_{name}", [x], "confirm", CONFIRM_SEEDS)
    cmd_report(args)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    from src.h2.learned import mu_alpha
    h = load_history()
    if not h:
        raise SystemExit("no history yet")
    lines = ["# Learned continuous mu and alpha (CMA-ES)", ""]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for phase, col in [("global", "tab:blue"), ("learned", "tab:red")]:
        rs = [r for r in h.values() if r["phase"] == phase]
        if rs:
            sc = [r["score"] for r in rs]
            ax[0].plot(range(1, len(sc) + 1), sc, "o", color=col, alpha=0.5, label=f"{phase} candidates")
            ax[0].plot(range(1, len(sc) + 1), np.maximum.accumulate(sc), "-", color=col, label=f"{phase} best so far")
            b = best_of(h, phase)
            lines += [f"**{phase} search** ({len(rs)} candidates, 500k generated transitions, 2 seeds): best {b['score']:.2f} "
                      f"at mean mu {b['mean_mu']:.3f}, mean alpha {b['mean_alpha']:.3f}, x = {b['x']}", ""]
    ax[0].set_xlabel("candidate")
    ax[0].set_ylabel("D4RL score (2 seeds)")
    ax[0].set_title("Search progress")
    ax[0].legend(fontsize=8)
    if any(r["phase"] == "learned" for r in h.values()):
        x = np.array(best_of(h, "learned")["x"])
        g = np.linspace(-0.5, 0.5, 41)
        S, Q = np.meshgrid(g, g)
        mu, al = mu_alpha(x, S.ravel(), Q.ravel())
        for a, v, t in [(ax[1], mu, "learned mu(S, Q)"), (ax[2], al, "learned alpha(S, Q)")]:
            im = a.imshow(v.reshape(S.shape), origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto")
            plt.colorbar(im, ax=a)
            a.set_xlabel("support S percentile")
            a.set_ylabel("quality Q percentile")
            a.set_title(t)
    fig.tight_layout()
    fig.savefig(OUT / "search.png", dpi=120)
    plt.close(fig)

    conf = {r["phase"][len("confirm_"):]: r for r in h.values() if r["phase"].startswith("confirm_")}
    if conf:
        lines += ["## Confirmation (1M generated transitions, 5 fresh seeds)", "",
                  "| setting | score | seeds | mean mu (sd) | mean alpha (sd) | edited reward | dyn MSE |", "|---|---|---|---|---|---|---|"]
        for n in ["default", "best_global", "best_learned"]:
            if n in conf:
                r = conf[n]
                lines.append(f"| {n} | {np.mean(r['scores']):.2f} ± {np.std(r['scores'], ddof=1):.2f} | "
                             f"{', '.join(f'{s:.1f}' for s in r['scores'])} | {r['mean_mu']:.3f} ({r['sd_mu']:.3f}) | "
                             f"{r['mean_alpha']:.3f} ({r['sd_alpha']:.3f}) | {r['oracle_r']:.1f} | {r['dyn_mse']:.4f} |")
        for a, b in [("best_learned", "best_global"), ("best_learned", "default"), ("best_global", "default")]:
            if a in conf and b in conf:
                p = ttest_ind(conf[a]["scores"], conf[b]["scores"], equal_var=False).pvalue
                lines.append(f"- {a} vs {b}: {np.mean(conf[a]['scores']) - np.mean(conf[b]['scores']):+.2f} (Welch p = {p:.3f})")
        lines += ["", "Decision rule: the learned map helps if best_learned beats best_global (p < 0.05)."]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        names = [n for n in ["default", "best_global", "best_learned"] if n in conf]
        for i, n in enumerate(names):
            v = conf[n]["scores"]
            ax.bar(i, np.mean(v), color=["grey", "tab:blue", "tab:red"][i], alpha=0.6)
            ax.scatter([i] * len(v), v, color="k", s=15, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        allv = [s for n in names for s in conf[n]["scores"]]
        ax.set_ylim(min(allv) - 2, max(allv) + 1)
        ax.set_ylabel("D4RL score (mean of last 5 evals)")
        ax.set_title("Confirmation: 5 fresh seeds, 1M generated transitions")
        fig.tight_layout()
        fig.savefig(OUT / "confirm_bars.png", dpi=120)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "report"])
    p.add_argument("--global_gens", type=int, default=3)
    p.add_argument("--learned_gens", type=int, default=7)
    a = p.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)

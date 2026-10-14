"""Experiment A report: learning curves, final scores, data quality per condition, verdict."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

ORDER = ["none", "uniform_0.3", "adaptive", "inverted", "uniform_0.5", "adaptive_wide", "inverted_wide"]
COLORS = {"none": "grey", "uniform_0.5": "tab:blue", "uniform_0.3": "tab:cyan", "adaptive": "tab:red",
          "inverted": "tab:orange",
          "adaptive_wide": "tab:purple", "inverted_wide": "tab:brown"}


def load_runs(out):
    rows = []
    for f in sorted((Path(out) / "runs").glob("*/evals.csv")):
        cond, seed = f.parent.name.rsplit("_seed", 1)
        rows.append(pd.read_csv(f).assign(condition=cond, seed=int(seed),
                                          done=(f.parent / "done.json").exists()))
    if not rows:
        raise SystemExit("No TD3+BC runs found yet.")
    return pd.concat(rows, ignore_index=True)


def data_quality(out):
    rows = []
    for name in ORDER[1:]:
        f = Path(out) / "data" / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            gain = (z["check_oracle_r"] - z["check_src_oracle_r"]) / np.abs(z["check_src_oracle_r"]).clip(1)
            rows.append(dict(condition=name, mean_mu=float(z["mu"].mean()), oracle_reward=float(z["check_oracle_r"].mean()),
                             source_reward=float(z["check_src_oracle_r"].mean()), rel_gain=float(gain.mean()),
                             dyn_mse=float(z["check_dyn_mse"].mean()),
                             frac_dyn_over_0085=float((z["check_dyn_mse"] > 0.0085).mean()),
                             shift=float(z["check_shift"].mean())))
    return pd.DataFrame(rows)


def report(out):
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    runs = load_runs(out)
    conds = [c for c in ORDER if c in set(runs.condition)]

    # final score per run = mean of the last 5 evaluations (finished runs only)
    fin = (runs[runs.done].sort_values("step").groupby(["condition", "seed"]).d4rl_score
           .apply(lambda s: s.tail(5).mean()).reset_index(name="final"))
    table = fin.groupby("condition").final.agg(["mean", "std", "count"]).reindex(conds)

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for c in conds:
        g = runs[runs.condition == c].groupby("step").d4rl_score
        m, s = g.mean().rolling(5, min_periods=1).mean(), g.std().fillna(0)
        ax[0].plot(m.index, m.values, color=COLORS[c], label=c)
        ax[0].fill_between(m.index, m - s, m + s, color=COLORS[c], alpha=0.15)
    ax[0].set_xlabel("TD3+BC training step")
    ax[0].set_ylabel("D4RL normalized score")
    ax[0].set_title("Learning curves (mean over seeds, ± std)")
    ax[0].legend(fontsize=8)
    for i, c in enumerate(conds):
        v = fin[fin.condition == c].final
        ax[1].bar(i, v.mean() if len(v) else np.nan, color=COLORS[c], alpha=0.6)
        ax[1].scatter(np.full(len(v), i), v, color="k", s=15, zorder=3)
    ax[1].set_xticks(range(len(conds)))
    ax[1].set_xticklabels(conds, rotation=15)
    ax[1].set_ylabel("final score (mean of last 5 evals)")
    lo = np.nanmin(fin.final) if len(fin) else 0
    ax[1].set_ylim(max(0, lo - 5), None)
    ax[1].set_title("Final scores (bars = mean, dots = seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / "rl_results.png", dpi=120)
    plt.close(fig)

    dq = data_quality(out)
    tests = {}
    a = fin[fin.condition == "adaptive"].final.values
    for c in ["uniform_0.3", "inverted", "uniform_0.5", "none"]:
        b = fin[fin.condition == c].final.values
        if len(a) >= 2 and len(b) >= 2:
            t = ttest_ind(a, b, equal_var=False)
            tests[f"adaptive_vs_{c}"] = {"diff": float(a.mean() - b.mean()), "welch_p": float(t.pvalue)}

    def beats(c):
        k = f"adaptive_vs_{c}"
        return k in tests and tests[k]["diff"] > 0 and tests[k]["welch_p"] < 0.05

    aw = fin[fin.condition == "adaptive_wide"].final.values
    for c in ["uniform_0.5", "inverted_wide"]:
        b = fin[fin.condition == c].final.values
        if len(aw) >= 2 and len(b) >= 2:
            t = ttest_ind(aw, b, equal_var=False)
            tests[f"adaptive_wide_vs_{c}"] = {"diff": float(aw.mean() - b.mean()), "welch_p": float(t.pvalue)}
    w = [tests.get(f"adaptive_wide_vs_{c}") for c in ["uniform_0.5", "inverted_wide"]]
    verdict_wide = ("INCOMPLETE" if None in w else
                    "SUPPORTED: adaptive_wide beats uniform_0.5 and inverted_wide" if all(x["diff"] > 0 and x["welch_p"] < 0.05 for x in w)
                    else "NOT SUPPORTED: adaptive_wide does not significantly beat both uniform_0.5 and inverted_wide")
    if beats("uniform_0.3") and beats("inverted"):
        verdict = "SUPPORTED: support-adaptive mu beats both same-average uniform mu and the inverted mapping"
    elif all(f"adaptive_vs_{c}" in tests for c in ["uniform_0.3", "inverted"]):
        verdict = "NOT SUPPORTED (yet): adaptive does not significantly beat both controls"
    else:
        verdict = "INCOMPLETE: need >= 2 finished seeds for adaptive, uniform_0.3 and inverted"

    summary = {"final_scores": table.round(2).reset_index().to_dict(orient="records"), "tests": tests,
               "data_quality": dq.round(4).to_dict(orient="records"), "verdict": verdict, "verdict_wide (mean mu 0.5)": verdict_wide}
    (out / "rl_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Experiment A: does support-adaptive mu improve TD3+BC?", "",
             "Same 161,291 source windows in every GTA condition; only the per-window mu differs. "
             "TD3+BC (CORL), batch 1024, 1M steps; final score = mean of the last 5 evaluations (10 episodes each).", "",
             f"**Verdict (mean mu 0.3):** {verdict}", "", f"**Verdict (mean mu 0.5, wide):** {verdict_wide}", "", "## Final D4RL scores", "",
             "| condition | mean | std | seeds |", "|---|---|---|---|"]
    lines += [f"| {c} | {r['mean']:.2f} | {r['std']:.2f} | {int(r['count'])} |" for c, r in table.iterrows()
              if np.isfinite(r["mean"])]
    lines += ["", "## Adaptive vs others (Welch t-test)", ""]
    lines += [f"- {k}: {v['diff']:+.2f} points, p = {v['welch_p']:.3f}" for k, v in tests.items()]
    if len(dq):
        lines += ["", "## Generated data quality (5,000-window check per condition)", "",
                  "| condition | mean mu | relative reward gain | dynamic MSE | share over 0.0085 | state shift |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r.condition} | {r.mean_mu:.3f} | {r.rel_gain:.3f} | {r.dyn_mse:.4f} | "
                  f"{r.frac_dyn_over_0085:.2f} | {r.shift:.2f} |" for r in dq.itertuples()]
    lines += ["", "Figure: `figures/rl_results.png`"]
    (out / "rl_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

"""Figure comparing support metrics: does each one tell windows apart, and does it predict physics error?

    python scripts/h8_plot.py      # -> outputs/h8_cluster_support/support_compare.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrootutils  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = "outputs/h8_cluster_support"
# (column, sign so that higher = more support, label, colour)
METRICS = [("s2_score", 1, "S2 recon. score", "#2a78d6"),
           ("s1_dist", -1, "S1 kNN", "#eb6834"),
           ("pg_ll", 1, "PG-Kmeans\npolicy likelihood", "#1baf7a"),
           ("caae_rec", 1, "CAAE\nreconstruction", "#eda100"),
           ("pg_ep_size", 1, "PG-Kmeans\nepisode-cluster size", "#e87ba4")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

pool = pd.read_csv(f"{OUT}/k8/pool_scores.csv")
for c, s, *_ in METRICS:
    pool[c + "_pct"] = (s * pool[c]).rank(pct=True)

sel = pd.read_csv("outputs/h2/selected.csv").merge(pool, on=["episode", "start"], suffixes=("_x", ""))
sw = pd.read_csv("outputs/h2/sweep.csv")
sel["dyn"] = sw[sw.mu <= 0.5].groupby("window").dyn_mse.mean().reindex(sel.window).values

fig = plt.figure(figsize=(14, 8.2), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.55, wspace=0.25)

# ---- row 1: windows of 12 random trajectories, one strip per trajectory ----
counts = pool.episode.value_counts()
eps = np.random.default_rng(1).choice(counts[counts >= 25].index.values, 12, replace=False)
for j, (c, _, label, col) in enumerate(METRICS):
    ax = fig.add_subplot(gs[0, j])
    for i, e in enumerate(eps):
        v = pool.loc[pool.episode == e, c + "_pct"].values
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v,
                   s=9, color=col, alpha=0.55, linewidths=0)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("12 random trajectories", color=MUTED)
    ax.set_title(label, color=INK, fontsize=10.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("support percentile of each window")
    distinct = pool[c].round(4).nunique()
    ax.text(0.5, -0.17, f"{distinct:,} distinct values", transform=ax.transAxes, ha="center",
            color=MUTED, fontsize=8.5)

# ---- row 2 left: rho with physics error, all windows and middle 80% ----
ax = fig.add_subplot(gs[1, :3])
rng = np.random.default_rng(0)
ys = np.arange(len(METRICS))[::-1]
for (c, s, label, col), y in zip(METRICS, ys):
    x, d = s * sel[c].values, sel.dyn.values
    rho = spearmanr(x, d)[0]
    boot = [spearmanr(x[i], d[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = (sel[c] > sel[c].quantile(0.1)) & (sel[c] < sel[c].quantile(0.9))
    rho_mid = spearmanr(x[m], d[m])[0]
    ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
    ax.scatter([rho], [y], s=70, color=col, edgecolors="#fcfcfb", linewidths=2, zorder=3)
    ax.scatter([rho_mid], [y - 0.28], s=40, facecolors="none", edgecolors=col, linewidths=1.6, zorder=3)
    ax.text(0.03, y, f"{rho:+.2f}", va="center", color=INK, fontsize=9.5)
    ax.text(0.03, y - 0.28, f"{rho_mid:+.2f}", va="center", color=MUTED, fontsize=8.5)
ax.set_yticks(ys)
ax.set_yticklabels([lab.replace("\n", " ") for _, _, lab, _ in METRICS], color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-0.55, 0.12)
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("Spearman ρ with simulator physics error (more negative = better support metric)")
ax.set_title("Does the score predict edit damage? (400 edited windows)", color=INK, fontsize=10.5, loc="left")
ax.scatter([], [], s=60, color=MUTED, label="all windows, with 95% CI")
ax.scatter([], [], s=40, facecolors="none", edgecolors=MUTED, label="middle 80% only (no extremes)")
ax.legend(loc="lower left", frameon=False, fontsize=8.5)

# ---- row 2 right: support percentile vs physics error, S2 vs episode-cluster size ----
for k, (c, s, label, col) in enumerate([METRICS[0], METRICS[4]]):
    ax = fig.add_subplot(gs[1, 3 + k])
    ax.scatter(sel[c + "_pct"], sel.dyn, s=10, color=col, alpha=0.6, linewidths=0)
    ax.set_yscale("log")
    ax.axhline(0.0085, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("support percentile")
    if k == 0:
        ax.set_ylabel("physics error (dyn. MSE)")
        ax.text(0.98, 0.0095, "θ", color=MUTED, ha="right", fontsize=9)
    ax.set_title(label.replace("\n", " "), color=INK, fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

fig.suptitle("Support metrics on halfcheetah-medium-v2: S2 and S1 rank windows and predict damage; "
             "trajectory-cluster scores are weaker or constant per trajectory", x=0.06, ha="left",
             color=INK, fontsize=12)
fig.savefig(f"{OUT}/support_compare.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"{OUT}/support_compare.png")

"""Has the reconstruction score (S2) collapsed? Distribution over the 20k pool, and agreement with the exact
ODE log-likelihood (S3, 100 windows). S1 shown alongside for comparison.

    python scripts/h8_s2_check.py      # -> outputs/h8_cluster_support/s2_check.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrootutils  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SIGMA_COLS = [("s2_err_sigma0.16", "σ = 0.16 (μ≈0.25)", "#86b6ef"),
              ("s2_err_sigma0.5", "σ = 0.5 (μ≈0.35)", "#2a78d6"),
              ("s2_err_sigma2.4", "σ = 2.4 (μ≈0.5)", "#104281")]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

pool = pd.read_csv("outputs/h2/pool.csv")
s3 = pool.dropna(subset=["s3_loglik"])
rng = np.random.default_rng(0)


def boot_ci(x, y, n=2000):
    b = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.percentile(b, [2.5, 97.5])


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(2, 3, figsize=(14, 8.4), facecolor="#fcfcfb")
fig.subplots_adjust(hspace=0.5, wspace=0.28)

# ---- row 1: S2 ----
ax = axs[0, 0]
x = pool.s2_score
ax.hist(x, bins=80, color=BLUE, edgecolor="#fcfcfb", linewidth=0.4)
q = x.quantile([0.05, 0.5, 0.95]).values
for v in q:
    ax.axvline(v, color=MUTED, lw=1, ls="--")
ax.set_title("S2 reconstruction score, 20,000 windows", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("s2_score (higher = better reconstructed)")
ax.set_ylabel("windows")
ax.text(0.02, 0.95, f"{x.round(4).nunique():,} distinct values\n5–95%: {q[0]:+.2f} … {q[2]:+.2f}\n"
        f"(≈{(q[2]-q[0])/x.std():.1f} std, no spike)", transform=ax.transAxes, va="top", color=INK, fontsize=9)
grid(ax)

ax = axs[0, 1]
bins = np.linspace(-3.2, 0.3, 90)
lines = []
for c, lab, col in SIGMA_COLS:
    e = np.log10(pool[c])
    ax.hist(e, bins=bins, color=col, alpha=0.75, edgecolor="none")
    r = pool[c].quantile(0.95) / pool[c].quantile(0.05)
    lines.append(f"{lab}: {r:.1f}× range")
ax.set_title("Raw reconstruction error at each noise level", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("log10 reconstruction error")
ax.set_ylabel("windows")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in SIGMA_COLS], lines, frameon=False, fontsize=8.5,
          loc="upper right", title="p95 / p5 of error", title_fontsize=8.5)
grid(ax)

ax = axs[0, 2]
rho, (lo, hi) = spearmanr(s3.s2_score, s3.s3_loglik)[0], boot_ci(s3.s2_score.values, s3.s3_loglik.values)
r_p = pearsonr(s3.s2_score, s3.s3_loglik)[0]
ax.scatter(s3.s2_score, s3.s3_loglik, s=22, color=BLUE, alpha=0.8, edgecolors="#fcfcfb", linewidths=0.8)
ax.set_title("S2 vs exact ODE log-likelihood (S3), 100 windows", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("s2_score")
ax.set_ylabel("S3: log p(window), prob.-flow ODE")
ax.text(0.03, 0.95, f"Spearman ρ = {rho:+.2f}  [95% CI {lo:+.2f}, {hi:+.2f}]\nPearson r = {r_p:+.2f}",
        transform=ax.transAxes, va="top", color=INK, fontsize=9)
grid(ax, "both")

# ---- row 2: S1 for comparison ----
ax = axs[1, 0]
x = pool.s1_dist
ax.hist(x, bins=80, color=ORANGE, edgecolor="#fcfcfb", linewidth=0.4)
q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
for v in q[[0, 2, 4]]:
    ax.axvline(v, color=MUTED, lw=1, ls="--")
share = ((x > 0.9 * q[2]) & (x < 1.1 * q[2])).mean()
ax.set_title("S1 kNN distance, 20,000 windows", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("s1_dist (lower = more support)")
ax.set_ylabel("windows")
ax.text(0.30, 0.95, f"{x.round(4).nunique():,} distinct values\n5–95%: {q[0]:.2f} … {q[4]:.2f} ({q[4]/q[0]:.1f}× range)\n"
        f"{share:.0%} of windows within ±10% of median", transform=ax.transAxes, va="top", color=INK, fontsize=9)
grid(ax)

ax = axs[1, 1]
for name, col, c, sgn in [("S2", BLUE, "s2_score", 1), ("S1", ORANGE, "s1_dist", -1)]:
    v = sgn * pool[c]
    z = (v - v.median()) / (v.quantile(0.75) - v.quantile(0.25))
    ax.hist(z.clip(-4, 4), bins=np.linspace(-4, 4, 81), histtype="step", lw=2, color=col, label=name)
ax.set_title("Shape compared on a common scale", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("(score − median) / IQR   (sign: higher = more support)")
ax.set_ylabel("windows")
ax.legend(frameon=False)
grid(ax)

ax = axs[1, 2]
rho1, (lo1, hi1) = spearmanr(-s3.s1_dist, s3.s3_loglik)[0], boot_ci(-s3.s1_dist.values, s3.s3_loglik.values)
ax.scatter(s3.s1_dist, s3.s3_loglik, s=22, color=ORANGE, alpha=0.8, edgecolors="#fcfcfb", linewidths=0.8)
ax.set_title("S1 vs exact ODE log-likelihood (S3), 100 windows", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("s1_dist")
ax.set_ylabel("S3: log p(window)")
ax.text(0.40, 0.95, f"Spearman ρ = {rho1:+.2f}  [95% CI {lo1:+.2f}, {hi1:+.2f}]", transform=ax.transAxes,
        va="top", color=INK, fontsize=9)
grid(ax, "both")

fig.savefig("outputs/h8_cluster_support/s2_check.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"S2-S3 spearman {rho:+.3f} [{lo:+.2f},{hi:+.2f}] pearson {r_p:+.3f}; S1-S3 {rho1:+.3f} [{lo1:+.2f},{hi1:+.2f}]")
print("S3 spread:", s3.s3_loglik.describe().round(1).to_dict())
for c, *_ in SIGMA_COLS:
    print(c, "p95/p5", round(pool[c].quantile(.95) / pool[c].quantile(.05), 1))

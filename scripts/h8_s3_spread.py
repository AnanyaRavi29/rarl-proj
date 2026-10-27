"""Spread of the exact ODE log-likelihood (S3) over its 100 windows.

    python scripts/h8_s3_spread.py      # -> outputs/h8_cluster_support/s3_spread.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrootutils  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

s = pd.read_csv("outputs/h2/pool.csv").dropna(subset=["s3_loglik"]).reset_index(drop=True)
y = s.s3_loglik
q = y.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
fig.subplots_adjust(wspace=0.3)

# ---- A: full range ----
ax = axs[0]
ax.hist(y, bins=40, color=BLUE, edgecolor=SURF, linewidth=0.6)
ax.axvspan(q[0.25], q[0.75], color=BLUE, alpha=0.12, lw=0)
ax.axvline(q[0.5], color=INK, lw=1.2)
for v in (q[0.05], q[0.95]):
    ax.axvline(v, color=MUTED, lw=1, ls="--")
low = s.nsmallest(1, "s3_loglik").iloc[0]
ax.annotate(f"episode {int(low.episode)}, step {int(low.start)}\n(also worst on S1 and S2)",
            xy=(low.s3_loglik, 1), xytext=(700, 12), color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_title("A. All 100 windows", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("S3 = log p(window) under the diffusion model  [nats]\n(higher = more likely = more support)")
ax.set_ylabel("number of windows")
ax.text(0.03, 0.97, f"range {y.min():,.0f} … {y.max():,.0f}\nmedian {q[0.5]:,.0f}\n"
        f"middle 50%: {q[0.25]:,.0f} … {q[0.75]:,.0f} (shaded)\n5–95% (dashed): {q[0.05]:,.0f} … {q[0.95]:,.0f}",
        transform=ax.transAxes, va="top", color=INK, fontsize=9)
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)

# ---- B: zoom on the bulk ----
ax = axs[1]
bulk = y[y > 1400]
ax.hist(bulk, bins=30, color=BLUE, edgecolor=SURF, linewidth=0.6)
ax.axvspan(q[0.25], q[0.75], color=BLUE, alpha=0.12, lw=0)
ax.axvline(q[0.5], color=INK, lw=1.2)
ax.set_title("B. Zoom: the 99 windows above 1,400", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("S3 [nats]")
ax.set_ylabel("number of windows")
d = q[0.75] - q[0.25]
ax.text(0.03, 0.97, f"middle 50% spans {d:,.0f} nats\n= ±{d / 2 / q[0.5]:.0%} of the median value\n"
        f"= a factor e^{d:,.0f} in probability\n= {d / 768:.2f} nats per number (32×24 window)",
        transform=ax.transAxes, va="top", color=INK, fontsize=9)
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)

# ---- C: where in the episode ----
ax = axs[2]
sc = ax.scatter(s.start, y, c=s.Q, cmap="Blues", vmin=-0.15, vmax=1, s=26, edgecolors=MUTED, linewidths=0.4)
ax.set_ylim(1350, 2300)
ax.text(0.98, 0.03, f"not shown: 1 window at {y.min():,.0f}", transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)
ax.set_title("C. Where in the trajectory (97 different episodes)", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("start step of the window within its episode (0–800)")
ax.set_ylabel("S3 [nats]")
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("quality percentile Q", color=INK)
cb.outline.set_visible(False)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)

fig.savefig("outputs/h8_cluster_support/s3_spread.png", dpi=160, bbox_inches="tight", facecolor=SURF)
print("outputs/h8_cluster_support/s3_spread.png")

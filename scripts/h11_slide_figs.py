"""Figures for the weekly-update deck (docs/h2_weekly/figures). Run in the gta env (MuJoCo rendering):

    python scripts/h11_slide_figs.py
"""
import json
import shutil
from pathlib import Path

import gym
import d4rl  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrootutils  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from src.h2.common import WindowData  # noqa: E402

FIG = Path("docs/h2_weekly/figures")
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948", "#8a8985"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved", name)


pool = pd.read_csv("outputs/h2/pool.csv")

# ---------- 1. S2 and S1 vs exact likelihood S3 ----------
s3 = pool.dropna(subset=["s3_loglik"])
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax, col, sgn, lab, c in [(axs[0], "s2_score", 1, "S2 reconstruction score  (right = more support)", BLUE),
                             (axs[1], "s1_dist", -1, "S1 neighbour distance  (left = more support)", ORANGE)]:
    ax.scatter(s3[col], s3.s3_loglik, s=28, color=c, edgecolors=SURF, linewidths=0.8)
    rho = spearmanr(sgn * s3[col], s3.s3_loglik)[0]
    ax.set_xlabel(lab)
    ax.set_ylabel("S3 exact log-likelihood\n(up = model finds window more likely)")
    ax.set_title(f"{col.split('_')[0].upper()} vs S3:  rank correlation = {rho:+.2f}")
    grid(ax, "both")
fig.tight_layout()
save(fig, "s3_agree.png")

# ---------- 2. mu by support, RL scores ----------
runs = Path("outputs/h2_rl_jax/runs")


def finals(cond):
    out = []
    for s in range(3):
        f = runs / f"{cond}_seed{s}" / "evals.csv"
        if f.exists():
            out.append(pd.read_csv(f).d4rl_score.iloc[-5:].mean())
    return np.array(out)


groups = [("average μ = 0.3", [("uniform_0.3", "same μ\nfor all", GRAY), ("adaptive", "support-\nmatched", BLUE),
                                 ("inverted", "reversed", ORANGE)]),
          ("average μ = 0.5", [("uniform_0.5", "same μ\nfor all", GRAY), ("adaptive_wide", "support-\nmatched", BLUE),
                                 ("inverted_wide", "reversed", ORANGE)])]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, sharey=True)
none = finals("none").mean()
for ax, (title, conds) in zip(axs, groups):
    for i, (c, lab, col) in enumerate(conds):
        v = finals(c)
        ax.bar(i, v.mean(), color=col, width=0.62)
        ax.scatter(np.full(len(v), i), v, color=INK, s=18, zorder=3)
        ax.text(i, max(v.max(), v.mean()) + 0.5, f"{v.mean():.1f}", ha="center", color=INK, fontsize=12, weight="bold")
    ax.axhline(none, color=MUTED, ls="--", lw=1)
    ax.text(-0.45, none - 0.9, f"no editing: {none:.1f}", ha="left", color=MUTED, fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _, lab, _ in conds])
    ax.set_ylim(45, 62)
    ax.set_title(title)
    grid(ax)
axs[0].set_ylabel("TD3+BC score (0 = random, 100 = expert)")
fig.tight_layout()
save(fig, "mu_support_rl.png")

# ---------- 3. alpha: score and physics error ----------
res = json.loads(Path("outputs/h2_alpha/results.json").read_text())
ALPHAS = ["1", "1.1", "1.3", "1.5", "1.8", "2"]
sc = {a: np.array([v["final"] for k, v in res.items() if k.startswith(f"alpha__alpha_{a}_seed")]) for a in ALPHAS}
phys, broken, speed = {}, {}, {}
for a in ALPHAS:
    d = np.load(f"outputs/h2_alpha/data/alpha_{a}.npz")
    phys[a], broken[a] = d["check_dyn_mse"].mean(), (d["check_dyn_mse"] > 0.0085).mean() * 100
    speed[a] = d["observations"][:, 8].mean()
x = [float(a) for a in ALPHAS]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURF)
ax = axs[0]
ax.plot(x, [sc[a].mean() for a in ALPHAS], color=BLUE, lw=2, marker="o", ms=8)
for a in ALPHAS:
    ax.scatter(np.full(len(sc[a]), float(a)), sc[a], color=INK, s=12, zorder=3, alpha=0.6)
    ax.text(float(a) + 0.045, sc[a].mean() - 0.2, f"{sc[a].mean():.1f}", ha="left", va="center", fontsize=11, weight="bold")
ax.axhline(none, color=MUTED, ls="--", lw=1)
ax.text(2.0, none - 1.4, f"no editing: {none:.1f}", ha="right", color=MUTED, fontsize=10)
ax.set_ylim(44, 62)
ax.set_xlabel("α (how much higher a return the edit asks for)")
ax.set_ylabel("TD3+BC score")
ax.set_title("A. Score rises, peaks at α = 1.3, then falls")
grid(ax, "both")
ax = axs[1]
ax.plot(x, [broken[a] for a in ALPHAS], color=RED, lw=2, marker="o", ms=8)
for a in ALPHAS:
    ax.annotate(f"{broken[a]:.0f}% broken\n{speed[a]:.1f} m/s", (float(a), broken[a]), xytext={"1": (0, -38), "1.1": (0, 12)}.get(a, (0, 12)),
                textcoords="offset points", ha="center", fontsize=10, color=INK)
ax.set_ylim(0, 110)
ax.set_xlabel("α (how much higher a return the edit asks for)")
ax.set_ylabel("edited windows that break physics (%)")
ax.set_title("B. Higher α: faster data, but more of it is broken")
grid(ax, "both")
fig.tight_layout()
save(fig, "alpha_curve.png")

# ---------- 4. CMA-ES illustration (simplified: sample, keep best, refit) ----------
rng = np.random.default_rng(4)


def f(p):
    x, y = p[..., 0], p[..., 1]
    return -((x - 2.2) ** 2 + 2.5 * (y - 1.6) ** 2 + 1.2 * (x - 2.2) * (y - 1.6))


gx, gy = np.meshgrid(np.linspace(-3, 4, 200), np.linspace(-3, 4, 200))
Z = f(np.stack([gx, gy], -1))
mean, cov = np.array([-1.8, -1.8]), np.eye(2) * 1.0
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5), facecolor=SURF)
gens = {0: axs[0], 3: axs[1], 9: axs[2]}
for g in range(10):
    pts = rng.multivariate_normal(mean, cov, 16)
    vals = f(pts)
    best = pts[np.argsort(vals)[-6:]]
    if g in gens:
        ax = gens[g]
        ax.contour(gx, gy, Z, levels=12, colors=GRID, linewidths=0.8)
        ax.scatter(2.2, 1.6, marker="*", s=260, color=YELLOW, edgecolors=INK, zorder=4, label="best setting (unknown to the search)")
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color=GRAY, label="tried this round")
        ax.scatter(best[:, 0], best[:, 1], s=40, color=BLUE, label="kept (best 6 of 16)")
        w, v = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        ax.add_patch(Ellipse(mean, 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, fill=False, color=BLUE, lw=1.8))
        ax.set_xlim(-3, 4)
        ax.set_ylim(-3, 4)
        ax.set_title(f"round {g + 1}")
        ax.set_xlabel("setting 1 (e.g. how μ depends on support)")
        if g == 0:
            ax.set_ylabel("setting 2")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
    new_mean = best.mean(0)
    step = new_mean - mean
    cov = 0.5 * cov + 0.5 * np.cov((best - mean).T, bias=True) + 0.3 * np.outer(step, step) + 0.01 * np.eye(2)
    mean = new_mean
fig.tight_layout()
save(fig, "cmaes_toy.png")

# ---------- 5. what each S2 quintile contains ----------
wd = WindowData("halfcheetah-medium-v2", 31, 0.99)
W = wd.windows(pool.episode.values, pool.start.values)[:, :31]
flip = (np.abs(W[:, :, 1]) > np.pi / 2).mean(1)
vel = W[:, :, 8].mean(1)
cat = np.select([flip > 0, pool.start < 30, vel < 3], ["flipped onto its back", "episode start (speeding up from rest)",
                                                      "slow (under 3 m/s)"], "normal running")
pool["cat"] = cat
pool["q5"] = pd.qcut(pool.S2, 5, labels=["1\n(least\nsupport)", "2", "3", "4", "5\n(most\nsupport)"])
tab = pd.crosstab(pool.q5, pool.cat, normalize="index") * 100
order = ["normal running", "episode start (speeding up from rest)", "slow (under 3 m/s)", "flipped onto its back"]
cols = [BLUE, YELLOW, ORANGE, RED]
fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=SURF)
left = np.zeros(5)
for c, col in zip(order, cols):
    v = tab[c].reindex(tab.index).fillna(0).values if c in tab else np.zeros(5)
    ax.barh(range(5), v, left=left, color=col, label=c, edgecolor=SURF, lw=1.5)
    for i, (l, w) in enumerate(zip(left, v)):
        if w > 4:
            ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", color="white" if col != YELLOW else INK, fontsize=11, weight="bold")
    left += v
ax.set_yticks(range(5))
ax.set_yticklabels(tab.index)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("share of the 4,000 windows in that group (%)")
ax.set_ylabel("support group (S2 quintile)")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=11)
fig.tight_layout()
save(fig, "quintile_content.png")
print((pd.crosstab(pool.q5, pool.cat)).to_string())

# ---------- 6. renders: windows at three support levels ----------
env = gym.make("halfcheetah-medium-v2")
sim = env.unwrapped
nq = sim.model.nq
STEPS = [0, 6, 12, 18, 24, 30]
sp = pool.sort_values("s2_score").reset_index(drop=True)
n = len(sp)
levels = [("least supported (lowest S2 of all 20,000)", sp.iloc[[0, 3]]),
          ("median support (50th percentile)", sp.iloc[[n // 2 - 1, n // 2 + 1]]),
          ("most supported (highest S2 of all 20,000)", sp.iloc[[n - 1, n - 3]])]
vlo, vhi = np.percentile(np.concatenate([e["obs"][:, 8] for e in wd.eps]), [10, 90])
fig = plt.figure(figsize=(15, 9.6), facecolor=SURF)
gs = fig.add_gridspec(10, 8, width_ratios=[1.25] + [1] * 6 + [1.45], height_ratios=[0.2] + [0.28, 1, 1] * 3, wspace=0.04, hspace=0.12)
for j, t in enumerate(STEPS):
    hx = fig.add_subplot(gs[0, 1 + j]); hx.axis('off'); hx.text(0.5, 0, f'step {t}', ha='center', fontsize=11, color=MUTED)
hx = fig.add_subplot(gs[0, 7]); hx.axis('off'); hx.text(0.5, 0, 'speed (m/s), shaded =\nusual speed range', ha='center', fontsize=10, color=MUTED)
r = 1
for title, g in levels:
    ax = fig.add_subplot(gs[r, :])
    ax.axis("off")
    ax.text(0, 0.0, title, fontsize=14, weight="bold", color=INK)
    r += 1
    for _, row in g.iterrows():
        win = wd.window(int(row.episode), int(row.start))
        st = win[:, :17].astype(np.float64)
        xpos = np.concatenate([[0.0], np.cumsum(st[:-1, nq - 1] * sim.dt)])
        a = fig.add_subplot(gs[r, 0])
        a.axis("off")
        a.text(0, 0.5, f"S2 percentile {row.S2:.2f}\nreward in window {row.q_return:.0f}\nmean speed {win[:-1, 8].mean():.1f} m/s",
               va="center", fontsize=10.5, color=INK)
        for j, t in enumerate(STEPS):
            sim.set_state(np.concatenate([[xpos[t]], st[t, :nq - 1]]), st[t, nq - 1:])
            a = fig.add_subplot(gs[r, 1 + j])
            a.imshow(sim.render(mode="rgb_array", width=200, height=200))
            a.axis("off")
        a = fig.add_subplot(gs[r, 7])
        a.axhspan(vlo, vhi, color="#cde2fb", lw=0)
        a.plot(win[:-1, 8], color=BLUE, lw=2)
        a.set_ylim(-2, 9)
        a.set_xticks([0, 15, 30])
        a.tick_params(labelsize=8)
        r += 1
save(fig, "support_levels_slide.png")

for src in ["outputs/h2_learn/search.png", "outputs/h2_learn/confirm_bars.png", "outputs/h8_render/window160_edits.png",
            "outputs/h8_render/window227_edits.png", "outputs/h8_render/why_gta.png"]:
    shutil.copy(src, FIG / Path(src).name)
print("copied existing figures")

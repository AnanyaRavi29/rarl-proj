"""Render 5 windows at each support level (S2, 20k pool): sparsest outliers, 25th, 50th, 75th percentile, best supported.

    python scripts/h8_support_levels.py      # gta env -> outputs/h8_render/support_levels.png
"""
from pathlib import Path

import gym
import d4rl  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrootutils  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from src.h2.common import WindowData  # noqa: E402

OUT = Path("outputs/h8_render")
OBS, H = 17, 31
STEPS = [0, 5, 10, 15, 20, 25, 30]
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

env = gym.make("halfcheetah-medium-v2")
sim = env.unwrapped
nq = sim.model.nq
wd = WindowData("halfcheetah-medium-v2", H, 0.99)
pool = pd.read_csv("outputs/h2/pool.csv").sort_values("s2_score").reset_index(drop=True)
n = len(pool)

groups = [("SPARSEST (lowest 5 of 20,000)", pool.iloc[:5]),
          ("25th percentile", pool.iloc[int(0.25 * n) - 2:int(0.25 * n) + 3]),
          ("50th percentile (median)", pool.iloc[int(0.50 * n) - 2:int(0.50 * n) + 3]),
          ("75th percentile", pool.iloc[int(0.75 * n) - 2:int(0.75 * n) + 3]),
          ("BEST SUPPORTED (highest 5 of 20,000)", pool.iloc[-5:][::-1])]


def frames(window, size=170):
    st = window[:, :OBS].astype(np.float64)
    x = np.concatenate([[0.0], np.cumsum(st[:-1, nq - 1] * sim.dt)])
    out = []
    for t in STEPS:
        sim.set_state(np.concatenate([[x[t]], st[t, :nq - 1]]), st[t, nq - 1:])
        out.append(sim.render(mode="rgb_array", width=size, height=size))
    return out


all_vel = np.concatenate([e["obs"][:, 8] for e in wd.eps])
vlo, vhi = np.percentile(all_vel, [10, 90])
rows = sum(len(g) for _, g in groups)
fig = plt.figure(figsize=(18, 1.75 * rows + 2.2), facecolor=SURF)
gs = fig.add_gridspec(rows + len(groups), 2 + len(STEPS), width_ratios=[1.35] + [1] * len(STEPS) + [1.5],
                      height_ratios=sum([[0.32] + [1] * len(g) for _, g in groups], []), wspace=0.03, hspace=0.12,
                      top=0.975, bottom=0.01)
r = 0
for title, g in groups:
    ax = fig.add_subplot(gs[r, :])
    ax.axis("off")
    ax.text(0, 0.1, title, color=INK, fontsize=13, weight="bold")
    r += 1
    for _, row in g.iterrows():
        win = wd.window(int(row.episode), int(row.start))
        ax = fig.add_subplot(gs[r, 0])
        ax.axis("off")
        ax.text(0, 0.5, f"S2 pct {row.S2:.3f}\ns2_score {row.s2_score:+.2f}\nS1 dist {row.s1_dist:.2f}\n"
                f"quality Q {row.Q:.2f}\nep {int(row.episode)}, step {int(row.start)}",
                color=INK, fontsize=8.3, va="center", family="monospace")
        for j, img in enumerate(frames(win)):
            a = fig.add_subplot(gs[r, 1 + j])
            a.imshow(img)
            a.axis("off")
            if r == 1:
                a.set_title(f"step {STEPS[j]}", color=MUTED, fontsize=9)
        a = fig.add_subplot(gs[r, -1])
        a.axhspan(vlo, vhi, color="#cde2fb", lw=0)
        a.plot(win[:-1, 8], color="#2a78d6", lw=2)
        a.set_ylim(-2, 10)
        a.set_xticks([0, 15, 30])
        a.tick_params(labelsize=7, colors=MUTED)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
        if r == 1:
            a.set_title("forward speed (m/s)\nshaded = usual 10–90%", color=MUTED, fontsize=8.5)
        r += 1
fig.suptitle("Windows at each level of reconstruction-score support (S2), 20,000-window pool", x=0.01, ha="left",
             color=INK, fontsize=14, y=0.99)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "support_levels.png", dpi=95, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "support_levels.png")
pd.concat([g.assign(level=t) for t, g in groups])[["level", "episode", "start", "S2", "s2_score", "s1_dist", "Q"]] \
    .to_csv(OUT / "support_levels.csv", index=False)

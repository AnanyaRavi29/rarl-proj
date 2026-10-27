"""See what support means: render quality-matched windows from every S2 quintile, and their GTA edits.

Outputs (outputs/h8_render/):
  gallery.png              10 windows (2 per S2 quintile, middle quality bin): filmstrip + numbers + velocity
  window<i>_edits.gif      original | mu 0.2 | 0.35 | 0.5 | 0.75, with physics error and true reward per panel
  window<i>_edits.png      the same as a still filmstrip (6 time steps per panel)

    python scripts/h8_render_support.py      # gta env (needs MuJoCo rendering)
"""
from pathlib import Path

import gym
import d4rl  # noqa: F401
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrootutils  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h8_render")
H2 = Path("outputs/h2")
OBS, ACT = 17, 6
STEPS = [0, 6, 12, 18, 24, 30]
GIF_MUS = [0.2, 0.35, 0.5, 0.75]
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

env = gym.make("halfcheetah-medium-v2")
sim = env.unwrapped
nq = sim.model.nq


def frames(window, size, label=None, steps=None):
    """Render each state of a window; x-position is rebuilt from forward velocity so the cheetah moves."""
    st = window[:, :OBS].astype(np.float64)
    x = np.concatenate([[0.0], np.cumsum(st[:-1, nq - 1] * sim.dt)])
    out = []
    for t in (steps if steps is not None else range(len(st))):
        sim.set_state(np.concatenate([[x[t]], st[t, :nq - 1]]), st[t, nq - 1:])
        img = Image.fromarray(sim.render(mode="rgb_array", width=size, height=size))
        if label:
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, size, 44], fill=(0, 0, 0))
            d.text((5, 3), f"{label}\nstep {t}", fill=(255, 255, 255))
        out.append(np.asarray(img))
    return out


sel = pd.read_csv(H2 / "selected.csv")
sw = pd.read_csv(H2 / "sweep.csv")
src = np.load(H2 / "sources.npy")
parts = {mu: np.load(H2 / "sweep_parts" / f"mu{mu:.2f}_d0.npz") for mu in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75]}
edited = {mu: p["edited"] for mu, p in parts.items()}
dyn = pd.DataFrame({mu: p["dyn_mse"] for mu, p in parts.items()})  # index = window 0..399, draw 0
rew = pd.DataFrame({mu: p["oracle_r"] for mu, p in parts.items()})

# ---- pick 2 windows per S2 quintile from the middle quality bin (quality held fixed) ----
mid = sel[sel.q_bin == 2].sort_values("S2")
picks = []
for b in range(5):
    g = mid[mid.s_bin == b]
    if b == 0:
        picks += list(g.window.iloc[:2])              # the two least supported
    elif b == 4:
        picks += list(g.window.iloc[-2:][::-1])       # the two best supported
    else:
        picks += list(g.window.iloc[[len(g) // 3, 2 * len(g) // 3]])
OUT.mkdir(parents=True, exist_ok=True)
info = sel.set_index("window").loc[picks]
info.assign(dyn_035=dyn.loc[picks, 0.35], dyn_05=dyn.loc[picks, 0.5]).to_csv(OUT / "picked.csv")

# ---- gallery ----
vel_all = np.concatenate([src[:, :-1, 8]])
lo, hi = np.percentile(vel_all, [10, 90])
fig = plt.figure(figsize=(17, 2.05 * len(picks)), facecolor=SURF)
gs = fig.add_gridspec(len(picks), 8, width_ratios=[1.55] + [1] * 6 + [1.5], wspace=0.04, hspace=0.25)
for r, w in enumerate(picks):
    row = info.loc[w]
    ax = fig.add_subplot(gs[r, 0])
    ax.axis("off")
    ax.text(0, 0.95, f"S2 quintile {int(row.s_bin) + 1} of 5\n", color=INK, fontsize=11, weight="bold", va="top")
    ax.text(0, 0.68, f"S2 percentile {row.S2:.2f}\nS1 distance {row.s1_dist:.2f}\nquality Q {row.Q:.2f}\n"
            f"episode {int(row.episode)}, step {int(row.start)}\n"
            f"physics error after edit:\n  μ=0.35: {dyn.loc[w, 0.35]:.4f}   μ=0.5: {dyn.loc[w, 0.5]:.4f}",
            color=INK, fontsize=8.6, va="top", family="monospace")
    for j, img in enumerate(frames(src[w], 200, steps=STEPS)):
        a = fig.add_subplot(gs[r, 1 + j])
        a.imshow(img)
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        if r == 0:
            a.set_title(f"step {STEPS[j]}", color=MUTED, fontsize=9)
    a = fig.add_subplot(gs[r, 7])
    a.axhspan(lo, hi, color="#cde2fb", lw=0)
    a.plot(src[w, :-1, 8], color="#2a78d6", lw=2)
    a.set_ylim(-1, 9)
    a.set_xticks([0, 15, 30])
    a.tick_params(labelsize=7, colors=MUTED)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    if r == 0:
        a.set_title("forward velocity (m/s)\nshaded = usual 10–90%", color=MUTED, fontsize=8.5)
fig.suptitle("Original windows from lowest (top) to highest (bottom) reconstruction-score support; "
             "all from the middle quality band", x=0.02, ha="left", color=INK, fontsize=13, y=1.0)
fig.savefig(OUT / "gallery.png", dpi=110, bbox_inches="tight", facecolor=SURF)
plt.close(fig)
print("saved", OUT / "gallery.png")

# ---- edits: GIF + still filmstrip for the 2 lowest and 2 highest support windows ----
for w in picks[:2] + picks[-2:]:
    row = info.loc[w]
    panels = [("original", src[w], None, rew.loc[w].iloc[0] if False else None)]
    labels = [f"ORIGINAL  S2 pct {row.S2:.2f}\nreward {row.src_oracle_r:.0f}"]
    wins = [src[w]]
    for mu in GIF_MUS:
        labels.append(f"mu={mu:g}  phys.err {dyn.loc[w, mu]:.4f}\nreward {rew.loc[w, mu]:.0f}")
        wins.append(edited[mu][w])
    clips = [frames(x, 220, lab) for x, lab in zip(wins, labels)]
    imageio.mimsave(OUT / f"window{w}_edits.gif", [np.concatenate(f, 1) for f in zip(*clips)], duration=0.09, loop=0)

    fig, axs = plt.subplots(len(wins), len(STEPS), figsize=(2.1 * len(STEPS), 2.25 * len(wins)), facecolor=SURF)
    for i, (x, lab) in enumerate(zip(wins, labels)):
        for j, img in enumerate(frames(x, 200, steps=STEPS)):
            axs[i, j].imshow(img)
            axs[i, j].set_xticks([]), axs[i, j].set_yticks([])
            if j == 0:
                axs[i, j].set_ylabel(lab.replace("\n", "\n"), fontsize=8.5, color=INK, rotation=0, ha="right", va="center")
            if i == 0:
                axs[i, j].set_title(f"step {STEPS[j]}", color=MUTED, fontsize=9)
    fig.suptitle(f"Window {w} (S2 quintile {int(row.s_bin) + 1}): original and GTA edits at increasing μ",
                 x=0.02, ha="left", color=INK, fontsize=12)
    fig.savefig(OUT / f"window{w}_edits.png", dpi=100, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved", OUT / f"window{w}_edits.gif")

"""Why is the D4RL score low, and what did GTA's data add? Speed of dataset vs GTA-generated windows, RL score
per alpha, and renders of original windows next to their GTA edits (mu 0.5; alpha 1.3 and 2.0).

    python scripts/h8_why_gta.py      # gta env -> outputs/h8_render/why_gta.png, why_gta_edits.png
"""
import pickle
from pathlib import Path

import gym
import d4rl  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyrootutils  # noqa: E402

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

from src.h2.common import WindowData  # noqa: E402

OUT = Path("outputs/h8_render")
H, OBS = 31, 17
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY, BLUE, ORANGE, RED = "#8a8985", "#2a78d6", "#eda100", "#e34948"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
ALPHAS = {"1": (49.14, GRAY), "1.3": (58.57, BLUE), "1.5": (58.10, ORANGE), "2": (52.61, RED)}
to_return = lambda s: s / 100 * 12415.18 - 280.18  # noqa: E731  D4RL normalisation (random -280, expert 12135)

eps = pickle.load(open("data/halfcheetah-medium-v2.pkl", "rb"))
data_v = np.concatenate([e["observations"][:, 8] for e in eps])
gen = {a: np.load(f"outputs/h2_alpha/data/alpha_{a}.npz") for a in ALPHAS}

# ---------------- figure 1: speed and score ----------------
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
bins = np.linspace(-2, 13, 151)
ax.hist(data_v, bins=bins, density=True, histtype="stepfilled", color=GRAY, alpha=0.35, label="original dataset (1M)")
for a, (_, col) in ALPHAS.items():
    ax.hist(gen[a]["observations"][:, 8], bins=bins, density=True, histtype="step", lw=2, color=col,
            label=f"GTA generated, α = {a}")
for x, lab in [(np.percentile(data_v, 99), "dataset 99th pct\n7.3 m/s"), (12.1, "expert\n(score 100)\n≈12 m/s")]:
    ax.axvline(x, color=MUTED, lw=1, ls="--")
    ax.text(x + 0.1, ax.get_ylim()[1] * 0.92, lab, color=MUTED, fontsize=8.5, va="top")
ax.set_xlabel("forward speed of the cheetah in each transition (m/s)  —  reward ≈ speed")
ax.set_ylabel("share of transitions (density)")
ax.set_title("A. The dataset never runs fast; GTA's edits do", color=INK, loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)

ax = axs[1]
for a, (score, col) in ALPHAS.items():
    g = gen[a]
    ax.scatter(g["observations"][:, 8].mean(), score, s=140, color=col, edgecolors=SURF, linewidths=2, zorder=3)
    ax.annotate(f"α={a}\nphysics err {g['check_dyn_mse'].mean():.4f}", (g["observations"][:, 8].mean(), score),
                xytext={"1": (10, 10), "1.3": (-10, 14), "1.5": (6, -30), "2": (-30, -34)}[a], textcoords="offset points",
                fontsize=8.5, color=INK)
ax.axhline(48.7, color=GRAY, lw=1, ls="--")
ax.text(7.2, 47.4, "no augmentation: 48.7", color=MUTED, fontsize=8.5)
ax.axhline(40.7, color=GRAY, lw=1, ls=":")
ax.text(5.0, 41.2, "average dataset episode: 40.7", color=MUTED, fontsize=8.5)
ax.set_xlim(4.8, 10.5)
ax.set_ylim(38, 62)
ax.set_xlabel("mean speed of GTA-generated transitions (m/s)")
ax.set_ylabel("TD3+BC D4RL score")
ax.set_title("B. Faster data helps — until physics breaks", color=INK, loc="left", fontsize=11)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "why_gta.png", dpi=140, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ---------------- figure 2: renders, original vs alpha 1.3 vs alpha 2 edit ----------------
env = gym.make("halfcheetah-medium-v2")
sim = env.unwrapped
nq = sim.model.nq
wd = WindowData("halfcheetah-medium-v2", H, 0.99)
full = dict(np.load("outputs/h2_rl/data/sources.npz"))
ix = np.sort(np.random.default_rng(2).choice(len(full["episode"]), 32258, replace=False))  # as in h2_alpha.py
src_e, src_s = full["episode"][ix], full["start"][ix]
STEPS = [0, 6, 12, 18, 24, 30]


def gen_window(a, i):
    g = gen[a]
    sl = slice(31 * i, 31 * i + 31)
    return np.concatenate([g["observations"][sl], g["next_observations"][sl][-1:]]), g["rewards"][sl]


def frames(st, size=170):
    st = st.astype(np.float64)
    x = np.concatenate([[0.0], np.cumsum(st[:-1, nq - 1] * sim.dt)])
    out = []
    for t in STEPS:
        sim.set_state(np.concatenate([[x[t]], st[t, :nq - 1]]), st[t, nq - 1:])
        out.append(sim.render(mode="rgb_array", width=size, height=size))
    return out


# windows from the 1000-window simulator check: typical gain and low physics error at alpha 1.3
c = gen["1.3"]
order = np.argsort(np.abs(c["check_oracle_r"] / c["check_src_oracle_r"] - 1.35) + 50 * c["check_dyn_mse"])
picks = [int(c["check_idx"][k]) for k in order[:2]]
rows = []
for i in picks:
    w = wd.window(int(src_e[i]), int(src_s[i]))
    rows.append((f"ORIGINAL\nep {src_e[i]}, step {src_s[i]}", w[:, :OBS], w[:-1, OBS + 6], GRAY))
    for a in ["1.3", "2"]:
        st, r = gen_window(a, i)
        rows.append((f"GTA edit  α = {a}", st, r, ALPHAS[a][1]))
fig = plt.figure(figsize=(16, 2.0 * len(rows)), facecolor=SURF)
gs = fig.add_gridspec(len(rows), 8, width_ratios=[1.2] + [1] * 6 + [1.6], wspace=0.03, hspace=0.15)
for r, (lab, st, rew, col) in enumerate(rows):
    ax = fig.add_subplot(gs[r, 0])
    ax.axis("off")
    ax.text(0, 0.5, f"{lab}\nreward {rew.sum():.0f} / 31 steps\nmean speed {st[:-1, 8].mean():.1f} m/s",
            color=INK, fontsize=9, va="center", family="monospace")
    for j, img in enumerate(frames(st)):
        a = fig.add_subplot(gs[r, 1 + j])
        a.imshow(img)
        a.axis("off")
        if r == 0:
            a.set_title(f"step {STEPS[j]}", color=MUTED, fontsize=9)
    a = fig.add_subplot(gs[r, 7])
    a.axhspan(np.percentile(data_v, 10), np.percentile(data_v, 90), color="#e4e3df", lw=0)
    a.plot(st[:-1, 8], color=col, lw=2)
    a.set_ylim(0, 12)
    a.set_xticks([0, 15, 30])
    a.tick_params(labelsize=7, colors=MUTED)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    if r == 0:
        a.set_title("forward speed (m/s)\ngray band = dataset 10–90%", color=MUTED, fontsize=8.5)
fig.suptitle("Original windows and their GTA edits (μ = 0.5): α = 1.3 runs faster with a longer stride; "
             "α = 2 pushes harder", x=0.01, ha="left", color=INK, fontsize=12.5, y=0.995)
fig.savefig(OUT / "why_gta_edits.png", dpi=100, bbox_inches="tight", facecolor=SURF)
print("saved", OUT / "why_gta.png", OUT / "why_gta_edits.png", "windows", picks)

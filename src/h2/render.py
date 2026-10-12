"""GIF of a source window next to its edits. The x-position is rebuilt from forward velocity so the
cheetah actually moves (the observation has no x), and the env's 'track' camera follows it."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _frames(env, window, obs_dim, act_dim, size, label):
    sim = env.unwrapped
    nq = sim.model.nq
    states = window[:, :obs_dim].astype(np.float64)
    x = np.concatenate([[0.0], np.cumsum(states[:-1, nq - 1] * sim.dt)])  # qvel[0] = forward velocity
    total_r = window[:-1, obs_dim + act_dim].sum()
    frames = []
    for t, s in enumerate(states):
        sim.set_state(np.concatenate([[x[t]], s[:nq - 1]]), s[nq - 1:])
        img = Image.fromarray(sim.render(mode="rgb_array", width=size, height=size))
        ImageDraw.Draw(img).text((6, 6), f"{label}\nreward {total_r:.0f}\nstep {t}", fill=(255, 255, 255))
        frames.append(np.asarray(img))
    return frames


def render_window(env, out, window, mus, draw, obs_dim, act_dim, size=240):
    import imageio
    out = Path(out)
    panels = [("original", np.load(out / "sources.npy")[window])]
    for mu in mus:
        f = out / "sweep_parts" / f"mu{mu:.2f}_d{draw}.npz"
        if not f.exists():
            print(f"skip mu={mu}: {f} not found")
            continue
        panels.append((f"mu={mu:g}", np.load(f)["edited"][window]))
    clips = [_frames(env, w, obs_dim, act_dim, size, lab) for lab, w in panels]
    frames = [np.concatenate(fr, axis=1) for fr in zip(*clips)]
    path = out / "figures" / f"window{window}_d{draw}.gif"
    path.parent.mkdir(exist_ok=True)
    imageio.mimsave(path, frames, duration=0.08)
    print(f"Saved {path}")

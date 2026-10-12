"""Source quality Q: discounted reward inside each H-step window."""
import numpy as np
import pandas as pd


def build_catalog(wd, gamma, max_start=800):
    """One row per source window (episode, start). Windows starting after max_start are dropped
    because their return-to-go (the GTA conditioning value) is cut short by the episode end."""
    kernel = gamma ** np.arange(wd.H)
    parts = []
    for e, ep in enumerate(wd.eps):
        n = min(len(ep["rew"]) - wd.H, max_start) + 1
        if n <= 0:
            continue
        q = np.convolve(ep["rew"].astype(np.float64), kernel[::-1], mode="valid")[:n]
        parts.append(pd.DataFrame(dict(episode=e, start=np.arange(n), q_return=q, rtg=ep["rtg"][:n])))
    return pd.concat(parts, ignore_index=True)

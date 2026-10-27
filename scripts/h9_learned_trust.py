"""Experiment 1: can a dynamics ensemble learned from the dataset replace the simulator as a physics-error
estimate for GTA-generated windows? (go/no-go for learned trust weighting, H6 without the oracle)

Per edited window, from the H5 ensemble (5 MLPs, (s, a) -> (delta s, r), trained on the original data):
  gap       mean over steps/dims of ((s'_gen - mean_k s'_k) / state_std)^2   (same units as the true dyn_mse)
  disagree  mean over steps/dims of var_k(s'_k) / state_std^2                 (epistemic: "never seen this")
Correlated with the TRUE simulator dyn_mse on
  (a) the H2 sweep: 400 windows x 8 mu x 2 draws, alpha 1.3
  (b) the alpha datasets: 1,000 simulator-checked windows each for alpha 1, 1.1, 1.3, 1.5, 1.8, 2 (mu 0.5)

    python scripts/h9_learned_trust.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

pyrootutils.setup_root(__file__, indicator=".aug-project-root", pythonpath=True, cwd=True)

OUT = Path("outputs/h9_learned_trust")
OBS, ACT, H, K = 17, 6, 31, 5
THETA = 0.0085
dev = "cuda" if torch.cuda.is_available() else "cpu"


def mlp():
    return nn.Sequential(nn.Linear(OBS + ACT, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, OBS + 1))


ck = torch.load("outputs/h5_vae_dyn/dyn.pt", map_location=dev)
ens = nn.ModuleList([mlp() for _ in range(K)]).to(dev)
ens.load_state_dict(ck["ens"])
ens.eval()
st = {k: v.to(dev) for k, v in ck["stats"].items()}

eps = pickle.load(open("data/halfcheetah-medium-v2.pkl", "rb"))
state_std = torch.tensor(np.concatenate([e["observations"] for e in eps]).std(0) + 1e-6, device=dev)


@torch.no_grad()
def score(s, a, s_next):
    """s, a, s_next: [N, T, dim] numpy. Returns per-window (gap, disagree)."""
    s, a, s_next = (torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (s, a, s_next))
    x = (torch.cat([s, a], -1) - st["in_mean"]) / st["in_std"]
    preds = torch.stack([m(x) for m in ens]) * st["out_std"] + st["out_mean"]  # [K, N, T, OBS+1]
    nxt = s[None] + preds[..., :OBS]
    gap = (((s_next - nxt.mean(0)) / state_std) ** 2).mean((1, 2))
    dis = (nxt.var(0) / state_std ** 2).mean((1, 2))
    return gap.cpu().numpy(), dis.cpu().numpy()


rows = []
# (a) H2 sweep
src_s2 = pd.read_csv("outputs/h2/selected.csv").set_index("window").s2_score
for f in sorted(Path("outputs/h2/sweep_parts").glob("mu*_d*.npz")):
    d = np.load(f)
    e = d["edited"]
    gap, dis = score(e[:, :H, :OBS], e[:, :H, OBS:OBS + ACT], e[:, 1:H + 1, :OBS])
    rows.append(pd.DataFrame(dict(set="H2 sweep (alpha 1.3)", mu=float(d["mu"]), alpha=1.3, true=d["dyn_mse"],
                                  gap=gap, disagree=dis, src_s2=src_s2.reindex(range(len(e))).values)))
# (b) alpha datasets
for a in ["1", "1.1", "1.3", "1.5", "1.8", "2"]:
    d = np.load(f"outputs/h2_alpha/data/alpha_{a}.npz")
    idx = d["check_idx"]
    sl = (idx[:, None] * H + np.arange(H)[None]).ravel()
    s = d["observations"][sl].reshape(len(idx), H, OBS)
    act = d["actions"][sl].reshape(len(idx), H, ACT)
    nx = d["next_observations"][sl].reshape(len(idx), H, OBS)
    gap, dis = score(s, act, nx)
    rows.append(pd.DataFrame(dict(set="alpha datasets (mu 0.5)", mu=0.5, alpha=float(a), true=d["check_dyn_mse"],
                                  gap=gap, disagree=dis, src_s2=np.nan)))
df = pd.concat(rows, ignore_index=True)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "scores.csv", index=False)


def stats(g, col):
    x, y = g[col].values, g.true.values
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    bad = y > THETA
    auc = roc_auc_score(bad, x if col != "src_s2" else -x) if 0 < bad.mean() < 1 else np.nan
    return dict(n=int(ok.sum()), spearman=spearmanr(x, y)[0], pearson_log=pearsonr(np.log(x - x.min() + 1e-9) if col == "src_s2" else np.log(x), np.log(y))[0],
                auc_over_theta=auc, share_over_theta=float(bad.mean()))


res = []
for (name, g) in [("H2 sweep, all mu", df[df.set.str.startswith("H2")]),
                  ("alpha sets, all alpha", df[df.set.str.startswith("alpha")])] + \
                 [(f"H2 sweep, mu={m:g}", g) for m, g in df[df.set.str.startswith("H2")].groupby("mu")] + \
                 [(f"alpha={a:g}", g) for a, g in df[df.set.str.startswith("alpha")].groupby("alpha")]:
    for col in ["gap", "disagree"] + (["src_s2"] if name.startswith("H2") else []):
        res.append(dict(subset=name, predictor=col, **stats(g, col)))
res = pd.DataFrame(res)
res.to_csv(OUT / "correlations.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
json.dump(res.to_dict("records"), open(OUT / "correlations.json", "w"), indent=1)

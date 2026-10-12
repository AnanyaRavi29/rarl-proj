"""Turn pool/sweep outputs into trend plots, statistics and a verdict (summary.md)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ----------------------------------------------------------------------------- pool

def plot_pool(pool, out):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax[0, 0].hist(pool.q_return, bins=60)
    ax[0, 0].set_title("Quality: discounted reward in window")
    ax[0, 1].hist(pool.s1_dist, bins=60)
    ax[0, 1].set_title("S1: kNN distance (lower = more support)")
    ax[0, 2].hist(pool.s2_score, bins=60)
    ax[0, 2].set_title("S2: recon score (higher = more support)")
    for a, (x, y) in zip(ax[1], [("Q", "S1"), ("Q", "S2"), ("S1", "S2")]):
        a.scatter(pool[x], pool[y], s=2, alpha=0.3)
        a.set_xlabel(f"{x} percentile")
        a.set_ylabel(f"{y} percentile")
        a.set_title(f"Spearman {x} vs {y} = {spearmanr(pool[x], pool[y])[0]:.3f}")
    fig.tight_layout()
    fig.savefig(out / "pool_overview.png", dpi=120)
    plt.close(fig)

    sub = pool.dropna(subset=["s3_loglik"])
    if len(sub):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, col in zip(ax, ["s2_score", "s1_dist"]):
            a.scatter(sub[col], sub.s3_loglik, s=12)
            a.set_xlabel(col)
            a.set_ylabel("S3 exact log-likelihood")
            a.set_title(f"Spearman = {spearmanr(sub[col], sub.s3_loglik)[0]:.3f} (n={len(sub)})")
        fig.tight_layout()
        fig.savefig(out / "pool_s3_check.png", dpi=120)
        plt.close(fig)


def pool_report(pool):
    """Checkpoint A: does support vary, is it independent of quality, do S1/S2/S3 agree?"""
    def spread(x):
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        return float((p90 - p10) / (abs(p50) + 1e-9))

    rep = {
        "n_pool": len(pool),
        "s1_dist_relative_spread_p90_p10": spread(pool.s1_dist),
        "spearman_Q_S1": float(spearmanr(pool.Q, pool.S1)[0]),
        "spearman_Q_S2": float(spearmanr(pool.Q, pool.S2)[0]),
        "spearman_S1_S2": float(spearmanr(pool.S1, pool.S2)[0]),
    }
    sub = pool.dropna(subset=["s3_loglik"])
    if len(sub):
        rep["spearman_S2_S3"] = float(spearmanr(sub.s2_score, sub.s3_loglik)[0])
        rep["spearman_S1_S3"] = float(spearmanr(-sub.s1_dist, sub.s3_loglik)[0])
    return rep


# ----------------------------------------------------------------------------- sweep

def load_sweep(out):
    sel = pd.read_csv(out / "selected.csv")
    rows = []
    for p in sorted((out / "sweep_parts").glob("*.npz")):
        z = np.load(p)
        rows.append(pd.DataFrame(dict(window=np.arange(len(z["gen_r"])), mu=float(z["mu"]), draw=int(z["draw"]),
                                      gen_r=z["gen_r"], oracle_r=z["oracle_r"], dyn_mse=z["dyn_mse"],
                                      shift=z["shift"])))
    if not rows:
        raise SystemExit("No sweep results yet (outputs/.../sweep_parts is empty).")
    df = pd.concat(rows, ignore_index=True).merge(sel, on="window")
    df["gain_abs"] = df.oracle_r - df.src_oracle_r
    df["gain_rel"] = df.gain_abs / df.src_oracle_r.abs().clip(lower=1.0)
    return sel, df


def best_mu(agg, thr):
    """mu*: highest gain among edits with dynamic MSE <= thr.  mu90: smallest mu reaching 90% of max gain."""
    rows = []
    for w, g in agg.groupby("window"):
        ok = g[g.dyn_mse <= thr]
        mu_star = float(ok.mu.values[np.argmax(ok.gain_rel.values)]) if len(ok) else np.nan
        gmax = g.gain_rel.max()
        mu90 = float(g.mu[g.gain_rel >= 0.9 * gmax].min()) if gmax > 0 else np.nan
        rows.append((w, mu_star, mu90))
    return pd.DataFrame(rows, columns=["window", "mu_star", "mu90"])


def _ols(q, s, y):
    X = np.column_stack([np.ones_like(q), q - 0.5, s - 0.5, (q - 0.5) * (s - 0.5)])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def regression(q, s, y, n_boot=2000, seed=0):
    """y ~ Q + S + Q*S on percentiles. Coefficients = change in y from lowest to highest percentile."""
    m = np.isfinite(y)
    q, s, y = q[m], s[m], y[m]
    rng = np.random.default_rng(seed)
    coef = _ols(q, s, y)
    boots = np.array([_ols(q[i], s[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    p_q = float((np.abs([_ols(rng.permutation(q), s, y)[1] for _ in range(n_boot)]) >= abs(coef[1])).mean())
    p_s = float((np.abs([_ols(q, rng.permutation(s), y)[2] for _ in range(n_boot)]) >= abs(coef[2])).mean())
    names = ["intercept", "Q", "S", "QxS"]
    return {"n": int(m.sum()),
            "coef": dict(zip(names, coef.round(4).tolist())),
            "ci95": {k: [float(a), float(b)] for k, a, b in zip(names, lo.round(4), hi.round(4))},
            "perm_p": {"Q": p_q, "S": p_s}}


def _quadrant(q, s):
    return np.where(q >= 0.5, "high Q", "low Q").astype(object) + " / " + np.where(s >= 0.5, "high S", "low S")


def _heat(ax, table, title, fmt=".2f", cmap="viridis"):
    im = ax.imshow(table.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in table.columns])
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7, color="w")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)


def analyze(out, support="S2", thr=0.017, n_boot=2000, seed=0):
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    sel, df = load_sweep(out)
    sel = sel.assign(S=sel[support])
    df = df.assign(S=df[support])
    df.to_csv(out / "sweep.csv", index=False)

    agg = df.groupby(["window", "mu"], as_index=False)[["gain_abs", "gain_rel", "dyn_mse", "shift", "oracle_r"]].mean()
    agg = agg.merge(sel[["window", "Q", "S", "q_bin", "s_bin", "src_oracle_r"]], on="window")
    best = best_mu(agg, thr).merge(sel[["window", "Q", "S", "q_bin", "s_bin"]], on="window")
    best["quadrant"] = _quadrant(best.Q.values, best.S.values)
    agg["quadrant"] = _quadrant(agg.Q.values, agg.S.values)
    best.to_csv(out / "best_mu.csv", index=False)
    mus = sorted(agg.mu.unique())
    nb = int(max(sel.q_bin.max(), sel.s_bin.max())) + 1
    bin_labels = [f"{int(100 * i / nb)}-{int(100 * (i + 1) / nb)}%" for i in range(nb)]

    # 1) heatmaps: mean gain / dynamic MSE vs mu, split by quality bin and support bin
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for r, (val, name, fmt) in enumerate([("gain_rel", "relative reward gain", ".2f"),
                                          ("dyn_mse", "dynamic MSE", ".3f")]):
        for c, (b, lab) in enumerate([("q_bin", "quality Q"), ("s_bin", f"support {support}")]):
            t = agg.pivot_table(index=b, columns="mu", values=val, aggfunc="mean")
            t.index = [bin_labels[int(i)] for i in t.index]
            _heat(ax[r, c], t, f"{name} by {lab} bin", fmt, "viridis" if val == "gain_rel" else "magma")
            ax[r, c].set_xlabel("mu")
            ax[r, c].set_ylabel(f"{lab} percentile bin")
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmaps.png", dpi=120)
    plt.close(fig)

    # 2) curves per quadrant
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for quad, g in agg.groupby("quadrant"):
        for a, val in zip(ax, ["gain_rel", "dyn_mse", "shift"]):
            s = g.groupby("mu")[val]
            m, e = s.mean(), s.sem()
            a.plot(m.index, m.values, marker="o", label=quad)
            a.fill_between(m.index, m - e, m + e, alpha=0.2)
    for a, t in zip(ax, ["relative reward gain", "dynamic MSE", "state shift (std units)"]):
        a.set_xlabel("mu")
        a.set_title(t)
    ax[1].axhline(thr, color="k", ls="--", lw=1, label=f"threshold {thr}")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "quadrant_curves.png", dpi=120)
    plt.close(fig)

    # 3) best mu vs Q and S
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    rng = np.random.default_rng(seed)
    for r, ycol in enumerate(["mu_star", "mu90"]):
        for c, xcol in enumerate(["Q", "S"]):
            b = best.dropna(subset=[ycol])
            jitter = rng.uniform(-0.01, 0.01, len(b))
            ax[r, c].scatter(b[xcol], b[ycol] + jitter, s=10, alpha=0.4)
            binned = b.groupby(np.minimum((b[xcol] * nb).astype(int), nb - 1))[ycol].mean()
            ax[r, c].plot((binned.index + 0.5) / nb, binned.values, "r-o", label="bin mean")
            rho = spearmanr(b[xcol], b[ycol])[0] if len(b) > 2 else np.nan
            ax[r, c].set_title(f"{ycol} vs {xcol}: Spearman {rho:.3f} (n={len(b)})")
            ax[r, c].set_xlabel(f"{xcol} percentile")
            ax[r, c].set_ylabel(ycol)
            ax[r, c].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_mu_vs_Q_S.png", dpi=120)
    plt.close(fig)

    # 4) the hypothesis map: mean mu* per (Q bin, S bin)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, ycol in zip(ax, ["mu_star", "mu90"]):
        t = best.pivot_table(index="q_bin", columns="s_bin", values=ycol, aggfunc="mean")
        t.index = [bin_labels[int(i)] for i in t.index]
        t.columns = [int(i) for i in t.columns]
        _heat(a, t, f"mean {ycol} per cell (rows: quality, cols: support bin)", ".2f", "coolwarm")
        a.set_xticklabels([bin_labels[int(i)] for i in t.columns])
        a.set_xlabel(f"support {support} percentile bin")
        a.set_ylabel("quality Q percentile bin")
    fig.tight_layout()
    fig.savefig(fig_dir / "best_mu_map.png", dpi=120)
    plt.close(fig)

    # 5) ceiling check: edited reward vs source reward
    cols = 4
    rows_ = -(-len(mus) // cols)
    fig, ax = plt.subplots(rows_, cols, figsize=(4 * cols, 3.6 * rows_), squeeze=False)
    lim = [agg.src_oracle_r.min(), max(agg.src_oracle_r.max(), agg.oracle_r.max())]
    for a, mu in zip(ax.ravel(), mus):
        g = agg[agg.mu == mu]
        a.scatter(g.src_oracle_r, g.oracle_r, s=6, alpha=0.5, c=g.Q, cmap="viridis")
        a.plot(lim, lim, "k--", lw=1)
        a.set_title(f"mu={mu:g}: rho={spearmanr(g.src_oracle_r, g.oracle_r)[0]:.2f}")
        a.set_xlabel("source reward (31 steps)")
        a.set_ylabel("edited reward")
    for a in ax.ravel()[len(mus):]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "ceiling_check.png", dpi=120)
    plt.close(fig)

    # statistics + verdict (decision rule fixed before the run)
    stats = {"support_metric": support, "dyn_mse_threshold": thr, "n_windows": int(sel.shape[0]),
             "mus": mus, "draws": int(df.draw.nunique())}
    for ycol in ["mu_star", "mu90"]:
        b = best.dropna(subset=[ycol])
        stats[ycol] = {
            "spearman_Q": float(spearmanr(b.Q, b[ycol])[0]),
            "spearman_S": float(spearmanr(b.S, b[ycol])[0]),
            "regression": regression(best.Q.values, best.S.values, best[ycol].values, n_boot, seed),
            "quadrant_mean": b.groupby("quadrant")[ycol].mean().round(3).to_dict(),
        }
        qm = stats[ycol]["quadrant_mean"]
        stats[ycol]["extreme_quadrant_diff"] = float(qm.get("low Q / high S", np.nan) - qm.get("high Q / low S", np.nan))

    def verdict(s):
        ci = s["regression"]["ci95"]
        s_ok, q_ok = ci["S"][0] > 0, ci["Q"][1] < 0
        big = s["extreme_quadrant_diff"] >= 0.1
        if s_ok and q_ok and big:
            return "SUPPORTED"
        if s_ok or q_ok:
            return "PARTLY SUPPORTED (" + ("S effect" if s_ok else "Q effect") + " only" + ("" if big else ", small") + ")"
        return "NOT SUPPORTED"

    stats["verdict_mu_star"] = verdict(stats["mu_star"])
    stats["verdict_mu90"] = verdict(stats["mu90"])
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    _write_summary(out, stats)
    print(json.dumps({k: stats[k] for k in ["verdict_mu_star", "verdict_mu90"]}, indent=2))
    print(f"Figures in {fig_dir}, summary in {out / 'summary.md'}")


def _write_summary(out, st):
    lines = ["# H2: does the best editing strength mu depend on quality Q and local support S?", "",
             f"Support metric: **{st['support_metric']}**, windows: {st['n_windows']}, mus: {st['mus']}, "
             f"draws: {st['draws']}, dynamic-MSE threshold: {st['dyn_mse_threshold']}", "",
             "Hypothesis: mu* rises with S and falls with Q. Decision rule: S coefficient CI > 0, "
             "Q coefficient CI < 0, and mu*(low Q/high S) - mu*(high Q/low S) >= 0.1.", ""]
    for ycol in ["mu_star", "mu90"]:
        s, r = st[ycol], st[ycol]["regression"]
        lines += [f"## {ycol}: {st['verdict_' + ycol]}", "",
                  f"- Spearman with Q: {s['spearman_Q']:.3f}, with S: {s['spearman_S']:.3f}",
                  f"- Regression (n={r['n']}), effect across the full percentile range:"]
        for k in ["Q", "S", "QxS"]:
            lines.append(f"  - {k}: {r['coef'][k]:+.3f}  CI95 [{r['ci95'][k][0]:+.3f}, {r['ci95'][k][1]:+.3f}]"
                         + (f"  permutation p={r['perm_p'][k]:.3f}" if k in r["perm_p"] else ""))
        lines.append(f"- Quadrant means: {s['quadrant_mean']}")
        lines.append(f"- Extreme-quadrant difference: {s['extreme_quadrant_diff']:+.3f}")
        lines.append("")
    lines += ["## Settings to keep in mind",
              "- Guidance uses the config's `temperature: 1.0` (w = 0), not the paper's w = 2.",
              "- The upstream network drops the return condition at random (p = 0.25) even while sampling; kept as-is.",
              "", "Figures: `figures/heatmaps.png`, `figures/quadrant_curves.png`, `figures/best_mu_vs_Q_S.png`, "
              "`figures/best_mu_map.png`, `figures/ceiling_check.png`, `pool_overview.png`."]
    (out / "summary.md").write_text("\n".join(lines))

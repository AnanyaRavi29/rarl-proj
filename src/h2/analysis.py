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


OUTCOMES = ["mu_star", "mu90", "mu_safe"]


def best_mu(agg, thr):
    """Per window:
    mu_star  highest gain among edits with dynamic MSE <= thr
    mu90     smallest mu reaching 90% of the window's max gain
    mu_safe  largest mu reached before dynamic MSE first exceeds thr (0 if already exceeded at the smallest mu)
    """
    rows = []
    for w, g in agg.sort_values("mu").groupby("window"):
        ok = g[g.dyn_mse <= thr]
        mu_star = float(ok.mu.values[np.argmax(ok.gain_rel.values)]) if len(ok) else np.nan
        gmax = g.gain_rel.max()
        mu90 = float(g.mu[g.gain_rel >= 0.9 * gmax].min()) if gmax > 0 else np.nan
        safe = g.dyn_mse.values <= thr
        n_safe = int(np.argmin(safe)) if (~safe).any() else len(safe)
        mu_safe = float(g.mu.values[n_safe - 1]) if n_safe > 0 else 0.0
        rows.append((w, mu_star, mu90, mu_safe))
    return pd.DataFrame(rows, columns=["window"] + OUTCOMES)


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


def analyze(out, support="S2", thr=0.017, n_boot=2000, seed=0, max_mu=None):
    out = Path(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    sel, df = load_sweep(out)
    if max_mu is not None:  # e.g. 0.5: only edits that keep part of the source window
        df = df[df.mu <= max_mu + 1e-9]
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
    fig, ax = plt.subplots(len(OUTCOMES), 2, figsize=(11, 4 * len(OUTCOMES)))
    rng = np.random.default_rng(seed)
    for r, ycol in enumerate(OUTCOMES):
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
    fig, ax = plt.subplots(1, len(OUTCOMES), figsize=(6.5 * len(OUTCOMES), 5))
    for a, ycol in zip(ax, OUTCOMES):
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

    # 6) support vs physics error, one panel per mu + correlation profile
    _plot_support_vs_physics(agg, sel, support, mus, thr, nb, fig_dir)

    # statistics + verdict (decision rule fixed before the run)
    stats = {"support_metric": support, "dyn_mse_threshold": thr, "max_mu": max_mu, "n_windows": int(sel.shape[0]),
             "mus": mus, "draws": int(df.draw.nunique())}
    for ycol in OUTCOMES:
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

    for ycol in OUTCOMES:
        stats["verdict_" + ycol] = verdict(stats[ycol])
    stats["correlation_profile"] = _correlation_profile(agg, sel, support).round(3).to_dict(orient="list")

    # 7) one-slide summary of all correlations
    _plot_correlation_summary(stats, best, agg, sel, support, nb, bin_labels, fig_dir)

    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    _write_summary(out, stats)
    print(json.dumps({k: stats["verdict_" + k] for k in OUTCOMES}, indent=2))
    print(f"Figures in {fig_dir}, summary in {out / 'summary.md'}")


def _correlation_profile(agg, sel, support):
    """Spearman correlations at every mu: support/quality vs physics error and vs reward gain."""
    other = "S1" if support != "S1" else "S2"
    a = agg.merge(sel[["window", other]], on="window") if other in sel else agg.assign(**{other: np.nan})
    rows = []
    for mu, g in a.groupby("mu"):
        rows.append(dict(mu=mu,
                         S_vs_dyn=spearmanr(g.S, g.dyn_mse)[0], Q_vs_dyn=spearmanr(g.Q, g.dyn_mse)[0],
                         S_vs_gain=spearmanr(g.S, g.gain_rel)[0], Q_vs_gain=spearmanr(g.Q, g.gain_rel)[0],
                         other_S_vs_dyn=spearmanr(g[other], g.dyn_mse)[0] if g[other].notna().all() else np.nan))
    return pd.DataFrame(rows)


def _plot_support_vs_physics(agg, sel, support, mus, thr, nb, fig_dir):
    cols = 4
    rows_ = -(-(len(mus) + 1) // cols)
    fig, ax = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.6 * rows_), squeeze=False)
    for a, mu in zip(ax.ravel(), mus):
        g = agg[agg.mu == mu]
        a.scatter(g.S, g.dyn_mse, s=6, alpha=0.35, c=g.Q, cmap="viridis")
        binned = g.groupby(np.minimum((g.S * nb).astype(int), nb - 1)).dyn_mse
        a.errorbar((binned.mean().index + 0.5) / nb, binned.mean().values, yerr=binned.sem().values,
                   fmt="r-o", lw=2, label="bin mean ± s.e.")
        a.axhline(thr, color="k", ls="--", lw=1)
        a.set_yscale("log")
        a.set_title(f"mu={mu:g}: Spearman(S, dyn MSE) = {spearmanr(g.S, g.dyn_mse)[0]:+.2f}")
        a.set_xlabel(f"support {support} percentile")
        a.set_ylabel("dynamic MSE (log)")
    ax.ravel()[0].legend(fontsize=7)
    prof = _correlation_profile(agg, sel, support)
    a = ax.ravel()[len(mus)]
    a.plot(prof.mu, prof.S_vs_dyn, "r-o", label=f"support {support}")
    if prof.other_S_vs_dyn.notna().all():
        a.plot(prof.mu, prof.other_S_vs_dyn, "r--s", alpha=0.6, label="other support metric")
    a.plot(prof.mu, prof.Q_vs_dyn, "b-o", label="quality Q")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("mu")
    a.set_ylabel("Spearman with dynamic MSE")
    a.set_title("Who predicts physics error?")
    a.legend(fontsize=7)
    for a in ax.ravel()[len(mus) + 1:]:
        a.axis("off")
    fig.suptitle("Edits of poorly supported windows break physics more (colour = quality)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "support_vs_physics.png", dpi=120)
    plt.close(fig)


def _plot_correlation_summary(stats, best, agg, sel, support, nb, bin_labels, fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))

    # (a) effect sizes with 95% CIs for every outcome
    y = 0
    ticks = []
    for ycol in OUTCOMES:
        r = stats[ycol]["regression"]
        for k, col in [("S", "tab:red"), ("Q", "tab:blue")]:
            lo, hi = r["ci95"][k]
            ax[0].errorbar(r["coef"][k], y, xerr=[[r["coef"][k] - lo], [hi - r["coef"][k]]], fmt="o", color=col,
                           capsize=4, label=("support S" if k == "S" else "quality Q") if ycol == OUTCOMES[0] else None)
            ticks.append((y, f"{ycol}  ({k})"))
            y += 1
        y += 0.5
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_yticks([t[0] for t in ticks])
    ax[0].set_yticklabels([t[1] for t in ticks])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("effect on mu, lowest -> highest percentile (95% CI)")
    ax[0].set_title("Effect of support and quality on the chosen mu")
    ax[0].legend(fontsize=8)

    # (b) mu_safe by support bin vs by quality bin
    for b, lab, col in [("s_bin", f"support {support}", "tab:red"), ("q_bin", "quality Q", "tab:blue")]:
        g = best.groupby(b).mu_safe
        ax[1].errorbar(g.mean().index, g.mean().values, yerr=g.sem().values, fmt="-o", color=col, capsize=4, label=lab)
    ax[1].set_xticks(range(nb))
    ax[1].set_xticklabels(bin_labels)
    ax[1].set_xlabel("percentile bin")
    ax[1].set_ylabel("mean mu_safe (± s.e.)")
    ax[1].set_title(f"How hard can a window be edited before physics breaks? (thr={stats['dyn_mse_threshold']})")
    ax[1].legend(fontsize=8)

    # (c) correlation profile across mu
    prof = _correlation_profile(agg, sel, support)
    ax[2].plot(prof.mu, prof.S_vs_dyn, "r-o", label="support vs physics error")
    ax[2].plot(prof.mu, prof.Q_vs_dyn, "b-o", label="quality vs physics error")
    ax[2].plot(prof.mu, prof.S_vs_gain, "r--s", alpha=0.6, label="support vs reward gain")
    ax[2].plot(prof.mu, prof.Q_vs_gain, "b--s", alpha=0.6, label="quality vs reward gain")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("mu")
    ax[2].set_ylabel("Spearman correlation")
    ax[2].set_title("Support predicts physics error; quality predicts reward gain")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_summary.png", dpi=120)
    plt.close(fig)


def _write_summary(out, st):
    lines = ["# H2: does the best editing strength mu depend on quality Q and local support S?", "",
             f"Support metric: **{st['support_metric']}**, windows: {st['n_windows']}, mus: {st['mus']}, "
             f"draws: {st['draws']}, dynamic-MSE threshold: {st['dyn_mse_threshold']}", "",
             "Hypothesis: mu* rises with S and falls with Q. Decision rule: S coefficient CI > 0, "
             "Q coefficient CI < 0, and mu*(low Q/high S) - mu*(high Q/low S) >= 0.1.", "",
             "Outcomes per window: `mu_star` = highest reward gain with dynamic MSE <= threshold; "
             "`mu90` = smallest mu reaching 90% of the max gain; `mu_safe` = largest mu before dynamic MSE "
             "first exceeds the threshold (added after the first main-run analysis, so exploratory there).", ""]
    for ycol in OUTCOMES:
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
    p = pd.DataFrame(st["correlation_profile"])
    lines += ["## Correlation profile (Spearman, per mu)", "",
              "| mu | support vs physics error | quality vs physics error | support vs gain | quality vs gain |",
              "|---|---|---|---|---|"]
    lines += [f"| {r.mu:g} | {r.S_vs_dyn:+.3f} | {r.Q_vs_dyn:+.3f} | {r.S_vs_gain:+.3f} | {r.Q_vs_gain:+.3f} |"
              for r in p.itertuples()]
    lines.append("")
    lines += ["## Settings to keep in mind",
              "- Guidance uses the config's `temperature: 1.0` (w = 0), not the paper's w = 2.",
              "- The upstream network drops the return condition at random (p = 0.25) even while sampling; kept as-is.",
              "", "Figures: `figures/heatmaps.png`, `figures/quadrant_curves.png`, `figures/best_mu_vs_Q_S.png`, "
              "`figures/best_mu_map.png`, `figures/ceiling_check.png`, `figures/support_vs_physics.png`, "
              "`figures/correlation_summary.png`, `pool_overview.png`."]
    (out / "summary.md").write_text("\n".join(lines))

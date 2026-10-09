"""M2 stage: Table 1, trial replication, hypothesis tests and descriptive plots.

M2 is trial-level inference on all randomized patients (as in the published
analysis), not model training, so it uses the full table; nothing here is
used to select or tune a model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equipoise import tracking
from equipoise.pipeline import prepare
from equipoise.preprocess.harmonize import MachineZScore
from equipoise.stats import inference, replication
from equipoise.stats.table1 import table1

MODULE = "m2"
EXPERIMENT = "m2-stats"
COLORS = {"Aflibercept": "#4C72B0", "Bevacizumab": "#DD8452", "Ranibizumab": "#55A868"}


def _with_harmonized_cst(df: pd.DataFrame, cfg) -> pd.DataFrame:
    f = cfg["features"]["cst"]
    harm = MachineZScore(cfg["preprocess"]["cst_min_machine_n"]).fit(df[[f["value"], f["machine"]]])
    return df.assign(X_cst_um_harm=harm.transform(df[[f["value"], f["machine"]]]).ravel())


def _plot_outcome_distributions(df, cfg, rep) -> None:
    arms = cfg["analysis"]["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in (
        (axes[0], "Y_va_change_wk52", "vision change at week 52 (letters)"),
        (axes[1], "Y_va_change_wk104", "vision change at week 104 (letters)"),
    ):
        data = [df.loc[df.treatment == a, col].dropna() for a in arms]
        parts = ax.violinplot(data, showmedians=True)
        for body, a in zip(parts["bodies"], arms, strict=True):
            body.set_facecolor(COLORS[a])
        ax.scatter(
            range(1, len(arms) + 1), [d.mean() for d in data], color="black", zorder=3, label="mean"
        )
        ax.set_xticks(
            range(1, len(arms) + 1), [f"{a}\n(n={len(d)})" for a, d in zip(arms, data, strict=True)]
        )
        ax.set_ylabel(lab)
        ax.axhline(0, color="grey", lw=0.8)
        ax.legend(loc="lower right")
    fig.suptitle(rep.title("Vision change by arm (complete cases)"))
    fig.tight_layout()
    rep.figure(fig, "outcome_by_arm")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (col, lab) in zip(
        axes,
        [
            ("X_va_letters", "baseline vision (letters)"),
            ("X_cst_um", "baseline CST (um, machine units)"),
            ("X_hba1c", "HbA1c (%)"),
        ],
        strict=True,
    ):
        for a in arms:
            ax.hist(
                df.loc[df.treatment == a, col].dropna(),
                bins=20,
                histtype="step",
                lw=1.5,
                color=COLORS[a],
                label=a,
                density=True,
            )
        ax.set_xlabel(lab)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.suptitle(rep.title("Baseline distributions by arm (randomization balance)"))
    fig.tight_layout()
    rep.figure(fig, "baseline_by_arm")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    pop = df[df[cfg["stats"]["yr1_population"]] == 1]
    bins = np.arange(pop["Y_n_inj_yr1"].min() - 0.5, pop["Y_n_inj_yr1"].max() + 1.5)
    for a in arms:
        ax.hist(
            pop.loc[pop.treatment == a, "Y_n_inj_yr1"],
            bins=bins,
            histtype="step",
            lw=1.5,
            color=COLORS[a],
            label=a,
        )
    ax.set_xlabel("injections in year 1 (patients with a week-52 visit)")
    ax.set_ylabel("patients (n)")
    ax.legend(fontsize=8)
    ax.set_title(rep.title("Year-1 injection counts by arm"))
    rep.figure(fig, "injections_by_arm")


def _plot_means(means: pd.DataFrame, cfg, rep) -> None:
    sub = means[means.outcome == "Y_va_change_wk52"]
    analyses = list(sub.analysis.unique())
    arms = cfg["analysis"]["arms"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    for i, a in enumerate(arms):
        r = sub[sub.arm == a].set_index("analysis").loc[analyses]
        x = np.arange(len(analyses)) + (i - 1) * width
        ax.errorbar(
            x,
            r["mean"],
            yerr=[r["mean"] - r["ci_low"], r["ci_high"] - r["mean"]],
            fmt="o",
            capsize=4,
            color=COLORS[a],
            label=a,
        )
    ax.set_xticks(
        np.arange(len(analyses)),
        [s.replace("week 52, ", "").replace(" (", "\n(") for s in analyses],
        fontsize=8,
    )
    ax.set_ylabel("mean letters gained at week 52 (95% CI)")
    ax.legend(fontsize=8)
    ax.set_title(rep.title("Replication: mean vision gain by arm and baseline-vision subgroup"))
    rep.figure(fig, "mean_gain_ci")


def _plot_correlation(df, cfg, rep) -> None:
    cols = cfg["stats"]["correlation_features"] + [cfg["analysis"]["primary_outcome"]]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.removeprefix("X_").removeprefix("Y_") for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson correlation (pairwise complete)")
    ax.set_title(rep.title("Correlation of baseline features and week-52 vision change"))
    rep.figure(fig, "correlation_heatmap")
    return corr


def _plot_va_vs_gain(df, cfg, rep) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    d = df.dropna(subset=["Y_va_change_wk52"])
    for a in cfg["analysis"]["arms"]:
        g = d[d.treatment == a]
        ax.scatter(g.X_va_letters, g.Y_va_change_wk52, s=8, alpha=0.35, color=COLORS[a])
        b1, b0 = np.polyfit(g.X_va_letters, g.Y_va_change_wk52, 1)
        xs = np.linspace(d.X_va_letters.min(), d.X_va_letters.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=COLORS[a], lw=2, label=f"{a} (slope {b1:.2f})")
    ax.axvline(
        cfg["trial"]["va_below_69_cutoff"] + 0.5, color="grey", ls="--", lw=1, label="20/50 cut-off"
    )
    ax.set_xlabel("baseline vision (ETDRS letters)")
    ax.set_ylabel("vision change at week 52 (letters)")
    ax.legend(fontsize=8)
    ax.set_title(rep.title("Gain vs baseline vision, with per-arm least-squares lines"))
    rep.figure(fig, "va_vs_gain")


def run(synthetic: bool) -> None:
    """Run M2 and log to MLflow experiment ``equipoise-m2-stats``."""
    sd = prepare(synthetic)
    cfg, df = sd.cfg, _with_harmonized_cst(sd.df, sd.cfg)
    rep = tracking.Reporter(MODULE, synthetic, cfg)
    outcome = cfg["analysis"]["primary_outcome"]
    with tracking.start_run(EXPERIMENT, synthetic=synthetic, data_path=sd.data_path, cfg=cfg):
        tracking.log_params(
            {"stats": {k: v for k, v in cfg["stats"].items() if k != "replication_targets"}}
        )
        rep.table(table1(df, cfg), "table1")

        res = replication.replicate(df, cfg)
        for name, t in res.items():
            rep.table(t, f"replication_{name}")
        check = replication.replication_check(res, cfg)
        rep.table(check, "replication_check")

        var = pd.concat(
            [inference.variance_tests(df, c, cfg) for c in (outcome, "Y_va_change_wk104")],
            ignore_index=True,
        )
        rep.table(var, "variance_tests")

        d = df[[outcome, "X_va_letters", "X_cst_um_harm"]].dropna()
        pc = inference.partial_correlation(
            d[outcome].to_numpy(), d.X_va_letters.to_numpy(), d.X_cst_um_harm.to_numpy()
        )
        pc["r_partial_matrix_check"] = inference.partial_corr_from_matrix(
            d[outcome].to_numpy(), d.X_va_letters.to_numpy(), d.X_cst_um_harm.to_numpy()
        )
        rep.table(
            pd.DataFrame(
                [{"x": outcome, "y": "X_va_letters", "controlling_for": "X_cst_um_harm", **pc}]
            ),
            "partial_correlation",
        )

        feats = (
            cfg["features"]["continuous"]
            + ["X_cst_um_harm"]
            + [c for c in cfg["features"]["binary"] if c != "X_va_below_69"]
        )
        mc = inference.multiple_correlation(df, outcome, feats)
        rep.table(pd.DataFrame([{"outcome": outcome, **mc}]), "multiple_correlation")

        coef_tabs, int_tests = [], []
        for moderator in ("X_va_letters", "X_va_below_69"):
            c, t = inference.interaction_regression(df, cfg, moderator)
            coef_tabs.append(c)
            int_tests.append(t)
        rep.table(pd.concat(coef_tabs, ignore_index=True), "interaction_regression")
        rep.table(pd.concat(int_tests, ignore_index=True), "interaction_tests")

        _plot_outcome_distributions(df, cfg, rep)
        _plot_means(res["means"], cfg, rep)
        _plot_correlation(df, cfg, rep)
        _plot_va_vs_gain(df, cfg, rep)

        m = res["means"]
        metrics = {
            f"mean_gain_wk52_{a.lower()}": float(
                m[(m.outcome == outcome) & (m.analysis == "week 52, all") & (m.arm == a)][
                    "mean"
                ].iloc[0]
            )
            for a in cfg["analysis"]["arms"]
        }
        it = pd.concat(int_tests)
        metrics |= {
            "p_interaction_va_letters": float(it.p_interaction.iloc[0]),
            "p_interaction_va_below_69": float(it.p_interaction.iloc[1]),
            "partial_r_gain_va_given_cst": pc["r_partial"],
            "multiple_R": mc["R"],
            "replication_all_match": float(check["match"].all()),
        }
        tracking.log_metrics(metrics)
    print(
        f"[stats] wrote {len(rep.written)} files; replication matches "
        f"{int(check['match'].sum())}/{len(check)} README targets"
    )

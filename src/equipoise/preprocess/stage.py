"""M1 stage: missing data, CST harmonization, outliers, encoding, bands."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from equipoise import tracking
from equipoise.pipeline import StageData, prepare
from equipoise.preprocess import missing, outliers
from equipoise.preprocess.discretize import discretize
from equipoise.preprocess.encode import build_preprocessor, onehot_sparse, transform_frame
from equipoise.preprocess.harmonize import MachineZScore

MODULE = "m1"
EXPERIMENT = "m1-preprocess"


def _missing_section(sd: StageData, rep: tracking.Reporter) -> dict[str, float]:
    miss = missing.missingness_table(sd.df)
    rep.table(miss, "missingness")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(miss) + 1.2))
    ax.barh(miss["column"], miss["n_missing"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("patients with missing value (n)")
    ax.set_title(rep.title(f"Missing values per column (N = {len(sd.df)})"))
    rep.figure(fig, "missingness")

    summary, per_imp = missing.compare_strategies(sd.df, sd.cfg)
    rep.table(summary, "missing_strategies")
    rep.table(per_imp, "mice_per_imputation")

    comps = list(summary["comparison"].unique())
    strategies = list(summary["strategy"].unique())
    fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 3.2), sharey=True)
    for ax, comp in zip(np.atleast_1d(axes), comps, strict=True):
        s = summary[summary["comparison"] == comp].set_index("strategy").loc[strategies]
        ypos = np.arange(len(s))
        ax.errorbar(
            s["estimate"],
            ypos,
            xerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
            fmt="o",
            capsize=4,
            color="#C44E52",
        )
        ax.axvline(0, color="grey", lw=1)
        ax.axvline(
            sd.cfg["analysis"]["mcid_letters"],
            color="grey",
            ls="--",
            lw=1,
            label=f"MCID = {sd.cfg['analysis']['mcid_letters']} letters",
        )
        ax.set_yticks(ypos, strategies)
        ax.set_xlabel("difference in week-52 vision change (letters, 95% CI)")
        ax.set_title(comp)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(rep.title("Primary arm differences under three missing-data strategies"))
    fig.tight_layout()
    rep.figure(fig, "missing_strategies")
    return {
        f"{r.strategy.split(' ')[0].lower()}_{r.comparison.split(' ')[0].lower()}_diff": r.estimate
        for r in summary.itertuples()
    }


def _cst_section(sd: StageData, rep: tracking.Reporter) -> None:
    f = sd.cfg["features"]["cst"]
    harm = MachineZScore(min_group_n=sd.cfg["preprocess"]["cst_min_machine_n"])
    harm.fit(sd.train[[f["value"], f["machine"]]])
    rep.table(harm.stats_table(), "cst_harmonization_fit")
    z = harm.transform(sd.df[[f["value"], f["machine"]]]).ravel()
    machine = sd.df[f["machine"]].fillna("Missing")
    before_after = pd.DataFrame({"machine": machine, "cst_um": sd.df[f["value"]], "cst_z": z})
    tab = (
        before_after.groupby("machine")
        .agg(
            n=("cst_um", "count"),
            cst_mean_um=("cst_um", "mean"),
            cst_sd_um=("cst_um", "std"),
            z_mean=("cst_z", "mean"),
            z_sd=("cst_z", "std"),
        )
        .reset_index()
    )
    rep.table(tab, "cst_by_machine")

    levels = [m for m in tab["machine"] if tab.set_index("machine").loc[m, "n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in (
        (axes[0], "cst_um", "CST, machine units (um)"),
        (axes[1], "cst_z", "harmonized CST (within-machine z-score)"),
    ):
        data = [before_after.loc[before_after.machine == m, col].dropna() for m in levels]
        ax.boxplot(
            data, tick_labels=[f"{m}\n(n={len(d)})" for m, d in zip(levels, data, strict=True)]
        )
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_title("Before harmonization")
    axes[1].set_title("After harmonization")
    fig.suptitle(rep.title("Baseline CST by OCT machine"))
    fig.tight_layout()
    rep.figure(fig, "cst_by_machine")


def _outlier_section(sd: StageData, rep: tracking.Reporter) -> dict[str, float]:
    res = outliers.detect(sd.df, sd.train_mask, sd.cfg)
    ocfg = sd.cfg["preprocess"]["outliers"]
    rep.table(res["counts"], "outlier_counts")
    rep.table(res["overlap"], "outlier_overlap")
    rep.table(res["per_feature"], "outlier_iqr_per_feature")
    res["flags"].to_csv(sd.derived_dir() / "m1_outlier_flags.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    kd = res["k_distance"]
    ax.plot(np.arange(len(kd)), kd, color="#4C72B0")
    ax.axhline(
        res["eps"],
        color="#C44E52",
        ls="--",
        label=f"eps = {res['eps']:.2f} ({ocfg['dbscan_eps_quantile']:.0%} quantile)",
    )
    ax.set_xlabel("training patients, sorted")
    ax.set_ylabel(
        f"distance to {sd.cfg['preprocess']['outliers']['dbscan_min_samples']}-th neighbour"
    )
    ax.set_title(rep.title("DBSCAN k-distance plot (scaled continuous baseline features)"))
    ax.legend()
    rep.figure(fig, "dbscan_k_distance")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ov = res["overlap"][res["overlap"]["flagged_by"] != "none"]
    ax.barh(ov["flagged_by"], ov["n"], color="#55A868")
    ax.invert_yaxis()
    ax.set_xlabel("patients (n)")
    ax.set_title(rep.title("Outlier flags: overlap between methods"))
    rep.figure(fig, "outlier_overlap")
    return {
        f"outliers_{m}": float(n)
        for m, n in zip(
            ["iqr", "isolation_forest", "dbscan_noise"], res["counts"]["n_flagged"], strict=True
        )
    } | {"dbscan_eps": res["eps"]}


def _encoding_section(sd: StageData, rep: tracking.Reporter) -> dict[str, float]:
    ct = build_preprocessor(sd.cfg).fit(sd.train)
    Z = transform_frame(ct, sd.df)
    rep.table(
        pd.DataFrame(
            {
                "encoded_feature": Z.columns,
                "train_mean": Z[sd.train_mask].mean().to_numpy(),
                "train_sd": Z[sd.train_mask].std().to_numpy(),
            }
        ),
        "encoded_features",
    )
    M, names = onehot_sparse(sd.df, sd.cfg)
    density = M.nnz / (M.shape[0] * M.shape[1])
    rep.table(
        pd.DataFrame(
            {
                "metric": [
                    "rows",
                    "one-hot columns",
                    "non-zeros",
                    "density",
                    "dense bytes",
                    "CSR bytes",
                ],
                "value": [
                    M.shape[0],
                    M.shape[1],
                    M.nnz,
                    round(density, 4),
                    M.shape[0] * M.shape[1] * 8,
                    M.data.nbytes + M.indices.nbytes + M.indptr.nbytes,
                ],
            }
        ),
        "onehot_sparse",
    )
    cont = sd.cfg["features"]["continuous"] + [sd.cfg["features"]["cst"]["value"]]
    rep.table(
        pd.DataFrame(
            {
                "feature": cont,
                "skewness": sd.df[cont].skew().to_numpy(),
                "log_transformed": [c in sd.cfg["features"]["log_transform"] for c in cont],
            }
        ),
        "skewness",
    )

    harm = MachineZScore(sd.cfg["preprocess"]["cst_min_machine_n"]).fit(
        sd.train[["X_cst_um", "X_oct_machine"]]
    )
    with_harm = sd.df.assign(X_cst_um_harm=harm.transform(sd.df[["X_cst_um", "X_oct_machine"]]))
    bands = discretize(with_harm, sd.cfg)
    counts = pd.concat(
        [
            bands[c]
            .value_counts(dropna=False)
            .rename_axis("band")
            .reset_index(name="n")
            .assign(feature=c)
            for c in bands.columns
        ]
    )[["feature", "band", "n"]]
    rep.table(counts, "bands")
    return {"n_encoded_features": float(Z.shape[1]), "onehot_density": density}


def run(synthetic: bool) -> None:
    """Run M1 and log everything to MLflow experiment ``equipoise-m1-preprocess``."""
    sd = prepare(synthetic)
    rep = tracking.Reporter(MODULE, synthetic, sd.cfg)
    with tracking.start_run(EXPERIMENT, synthetic=synthetic, data_path=sd.data_path, cfg=sd.cfg):
        tracking.log_params({"preprocess": sd.cfg["preprocess"], "features": sd.cfg["features"]})
        metrics = {}
        metrics |= _missing_section(sd, rep)
        _cst_section(sd, rep)
        metrics |= _outlier_section(sd, rep)
        metrics |= _encoding_section(sd, rep)
        tracking.log_metrics(metrics)
    print(f"[preprocess] wrote {len(rep.written)} files to {rep.written[0].parent.parent}")

"""M5 stage: outcome models for vision change, responders, laser and injection counts.

All models are tuned by cross-validation on the training 80 % (saved folds)
and evaluated once on the held-out test set. Each model is a nested MLflow
run under the stage run. The arm-mean baseline shows how much the baseline
features add beyond knowing the randomized arm.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_validate

from equipoise import tracking
from equipoise.pipeline import StageData, prepare
from equipoise.preprocess.encode import transform_frame
from equipoise.supervised import knn_timing, models
from equipoise.supervised.design import (
    ArmMeanRegressor,
    ArmRateClassifier,
    full_rank_preprocessor,
    outcome_preprocessor,
    predefined_cv,
)
from equipoise.supervised.scratch import IRLSLogisticRegression, normal_equations

MODULE = "m5"
EXPERIMENT = "m5-supervised"
REG_CV = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
REG_KEYS = {"CV_RMSE": ("rmse", -1), "CV_MAE": ("mae", -1), "CV_R2": ("r2", 1)}
CLF_CV = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
CLF_KEYS = {"CV_Accuracy": ("accuracy", 1), "CV_F1": ("f1", 1), "CV_AUC": ("roc_auc", 1)}
CNT_CV = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "dev": "neg_mean_poisson_deviance",
}
CNT_KEYS = {"CV_RMSE": ("rmse", -1), "CV_MAE": ("mae", -1), "CV_Poisson_deviance": ("dev", -1)}


def _rows(
    sd: StageData, mask_col: str | None, outcome: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test rows for one target (population filter and observed outcome)."""
    test_ids = set(sd.splits.test_ids)
    df = sd.df
    if mask_col:
        df = df[df[mask_col] == 1]
    if outcome:
        df = df[df[outcome].notna()]
    is_test = df["pt_id"].isin(test_ids)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


# ---------------------------------------------------------------- regression
def _regression(sd: StageData, rep: tracking.Reporter) -> tuple[pd.DataFrame, dict]:
    cfg, y_col = sd.cfg, sd.cfg["analysis"]["primary_outcome"]
    tr, te = _rows(sd, None, y_col)
    cv = predefined_cv(tr, sd.splits)
    pre = outcome_preprocessor(cfg, tr)
    rows, preds = [], {}
    base = ArmMeanRegressor()
    cvr = cross_validate(base, tr, tr[y_col], cv=cv, scoring=REG_CV)
    base.fit(tr, tr[y_col])
    rows.append(
        {
            "Model": "Baseline (arm mean)",
            **models.summarize_cv(cvr, REG_KEYS),
            **{
                f"Test_{k}": v
                for k, v in models.regression_metrics(te[y_col], base.predict(te)).items()
            },
            "best_params": "",
        }
    )
    for name, (est, grid) in models.regression_models(cfg).items():
        f = models.tune(
            name,
            est,
            grid,
            pre,
            tr,
            tr[y_col].to_numpy(),
            cv,
            cfg["supervised"]["cv_scoring_regression"],
            REG_CV,
        )
        yhat = f.model.predict(te)
        preds[name] = yhat
        test = models.regression_metrics(te[y_col].to_numpy(), yhat)
        row = {
            "Model": models.REG_LABELS[name],
            **models.summarize_cv(f.cv_results, REG_KEYS),
            **{f"Test_{k}": v for k, v in test.items()},
            "best_params": str(f.best_params),
        }
        rows.append(row)
        tracking.child_run(
            f"regression/{name}",
            f.best_params,
            {k: v for k, v in row.items() if isinstance(v, float)},
        )
    table = pd.DataFrame(rows)
    rep.table(table, "regression_comparison")

    best = min(preds, key=lambda n: table.set_index("Model").loc[models.REG_LABELS[n], "CV_RMSE"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    t = table.set_index("Model")
    ypos = np.arange(len(t))
    axes[0].barh(ypos - 0.2, t["CV_RMSE"], 0.4, label="5-fold CV (training)", color="#4C72B0")
    axes[0].barh(ypos + 0.2, t["Test_RMSE"], 0.4, label="held-out test", color="#DD8452")
    axes[0].set_yticks(ypos, t.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (letters)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("RMSE by model (lower is better)")
    axes[1].scatter(te[y_col], preds[best], s=10, alpha=0.6)
    lim = [min(te[y_col].min(), preds[best].min()), max(te[y_col].max(), preds[best].max())]
    axes[1].plot(lim, lim, color="grey", ls="--")
    axes[1].set_xlabel("observed week-52 vision change (letters)")
    axes[1].set_ylabel("predicted (letters)")
    axes[1].set_title(f"{models.REG_LABELS[best]} on the held-out test set")
    fig.suptitle(rep.title("Regression of week-52 vision change"))
    fig.tight_layout()
    rep.figure(fig, "regression_comparison")

    # normal equations vs sklearn on a full-rank design
    fr = full_rank_preprocessor(cfg).fit(tr)
    Xf = fr.transform(tr)
    b = normal_equations(Xf, tr[y_col].to_numpy())
    lr = LinearRegression().fit(Xf, tr[y_col].to_numpy())
    lst = np.linalg.lstsq(
        np.column_stack([np.ones(len(Xf)), Xf]), tr[y_col].to_numpy(), rcond=None
    )[0]
    rep.table(
        pd.DataFrame(
            [
                {
                    "n": len(Xf),
                    "p_with_intercept": Xf.shape[1] + 1,
                    "max_abs_diff_vs_sklearn": float(
                        np.abs(b - np.r_[lr.intercept_, lr.coef_]).max()
                    ),
                    "max_abs_diff_vs_lstsq": float(np.abs(b - lst).max()),
                    "condition_number_XtX": float(
                        np.linalg.cond(
                            np.column_stack([np.ones(len(Xf)), Xf]).T
                            @ np.column_stack([np.ones(len(Xf)), Xf])
                        )
                    ),
                }
            ]
        ),
        "normal_equations_check",
    )
    coef = pd.DataFrame(
        {
            "term": ["intercept", *fr.get_feature_names_out()],
            "coef_normal_eq": b,
            "coef_sklearn": np.r_[lr.intercept_, lr.coef_],
        }
    )
    rep.table(coef, "ols_coefficients")

    # kNN neighbour-search structures
    pre_fit = outcome_preprocessor(cfg, tr).fit(tr)
    Ztr, Zte = transform_frame(pre_fit, tr).to_numpy(), transform_frame(pre_fit, te).to_numpy()
    kt = cfg["supervised"]["knn_timing"]
    timing = knn_timing.compare(Ztr, Zte, kt["algorithms"], kt["n_neighbors"], kt["repeats"])
    rep.table(timing, "knn_tree_timing")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(timing))
    ax.bar(x - 0.2, timing.fit_ms_median, 0.4, label="fit")
    ax.bar(x + 0.2, timing.query_ms_median, 0.4, label=f"query {len(Zte)} test patients")
    ax.set_xticks(x, timing.algorithm)
    ax.set_ylabel("median time (ms)")
    ax.legend()
    ax.set_title(rep.title(f"kNN search structures ({Ztr.shape[1]} dims, k={kt['n_neighbors']})"))
    rep.figure(fig, "knn_tree_timing")
    return table, {"best_regression_model": best}


# ------------------------------------------------------------ classification
def _classification(sd: StageData, rep: tracking.Reporter, target: str) -> pd.DataFrame:
    cfg, s = sd.cfg, sd.cfg["supervised"]
    y_out = cfg["analysis"]["primary_outcome"]
    if target == "gain_ge_15":
        tr, te = _rows(sd, None, y_out)
        thr = cfg["analysis"]["responder_threshold_letters"]
        ytr, yte = (tr[y_out] >= thr).astype(int), (te[y_out] >= thr).astype(int)
        title = f"Responder (gain >= {thr} letters at week 52)"
    else:
        tr, te = _rows(sd, s["laser_population"], None)
        ytr, yte = tr["Y_laser_yr1"].astype(int), te["Y_laser_yr1"].astype(int)
        title = "Rescue laser by year 1"
    cv = predefined_cv(tr, sd.splits)
    pre = outcome_preprocessor(cfg, tr)
    rows, probs = [], {}
    base = ArmRateClassifier()
    cvr = cross_validate(base, tr, ytr, cv=cv, scoring=CLF_CV)
    base.fit(tr, ytr)
    probs["Baseline (arm rate)"] = base.predict_proba(te)[:, 1]
    rows.append(
        {
            "Model": "Baseline (arm rate)",
            "class_weight": "none",
            **models.summarize_cv(cvr, CLF_KEYS),
            **models.classification_metrics(yte.to_numpy(), probs["Baseline (arm rate)"]),
        }
    )
    for cw in (None, "balanced"):
        for name, (est, grid) in models.classification_models(cfg, cw).items():
            if cw and not any(k in est.get_params() for k in ("class_weight",)):
                continue  # kNN / naive Bayes have no class weights
            f = models.tune(
                name, est, grid, pre, tr, ytr.to_numpy(), cv, s["cv_scoring_classification"], CLF_CV
            )
            p = f.model.predict_proba(te)[:, 1]
            label = name + (" (balanced)" if cw else "")
            probs[label] = p
            row = {
                "Model": label,
                "class_weight": cw or "none",
                **models.summarize_cv(f.cv_results, CLF_KEYS),
                **models.classification_metrics(yte.to_numpy(), p),
                "best_params": str(f.best_params),
            }
            rows.append(row)
            tracking.child_run(
                f"{target}/{label}",
                f.best_params,
                {
                    k: v
                    for k, v in row.items()
                    if isinstance(v, float | int) and not isinstance(v, bool)
                },
            )
    table = pd.DataFrame(rows)
    table.insert(0, "target", target)
    table["n_train"], table["n_test"], table["test_prevalence"] = len(tr), len(te), yte.mean()
    rep.table(table, f"classification_{target}")

    unweighted = [m for m in probs if "(balanced)" not in m]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in unweighted:
        fpr, tpr, _ = roc_curve(yte, probs[m])
        auc = table.set_index("Model").loc[m, "AUC"]
        axes[0].plot(fpr, tpr, label=f"{m} (AUC {auc:.2f})", ls="--" if "Baseline" in m else "-")
        prec, rec, _ = precision_recall_curve(yte, probs[m])
        axes[1].plot(rec, prec, label=m, ls="--" if "Baseline" in m else "-")
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.8)
    axes[0].set_xlabel("1 - specificity (false-positive rate)")
    axes[0].set_ylabel("sensitivity (recall)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("ROC curves (held-out test)")
    axes[1].axhline(yte.mean(), color="grey", lw=0.8, label=f"prevalence {yte.mean():.2f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Precision-recall curves (held-out test)")
    fig.suptitle(rep.title(title))
    fig.tight_layout()
    rep.figure(fig, f"roc_pr_{target}")

    show = [m for m in probs if m != "Baseline (arm rate)"]
    ncol = 4
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    for ax, m in zip(np.ravel(axes), show, strict=False):
        ConfusionMatrixDisplay.from_predictions(
            yte, (probs[m] >= 0.5).astype(int), ax=ax, colorbar=False, labels=[0, 1]
        )
        ax.set_title(m, fontsize=8)
    for ax in np.ravel(axes)[len(show) :]:
        ax.axis("off")
    fig.suptitle(rep.title(f"{title}: confusion matrices at threshold 0.5 (test)"))
    fig.tight_layout()
    rep.figure(fig, f"confusion_{target}")
    return table


def _irls_check(sd: StageData, rep: tracking.Reporter) -> dict[str, float]:
    cfg, y_out = sd.cfg, sd.cfg["analysis"]["primary_outcome"]
    tr, _ = _rows(sd, None, y_out)
    y = (tr[y_out] >= cfg["analysis"]["responder_threshold_letters"]).astype(int).to_numpy()
    X = full_rank_preprocessor(cfg).fit_transform(tr)
    ir = IRLSLogisticRegression(
        cfg["supervised"]["irls"]["max_iter"], cfg["supervised"]["irls"]["tol"], l2=0.0
    ).fit(X, y)
    sm_fit = sm.Logit(y, sm.add_constant(X)).fit(disp=0, method="newton")
    ours = np.r_[ir.intercept_, ir.coef_.ravel()]
    diff = float(np.abs(ours - sm_fit.params).max())
    rep.table(
        pd.DataFrame(
            [
                {
                    "n": len(X),
                    "p_with_intercept": X.shape[1] + 1,
                    "irls_iterations": ir.n_iter_,
                    "max_abs_coef_diff_vs_statsmodels": diff,
                    "loglik_irls": ir.loglik_path_[-1],
                    "loglik_statsmodels": sm_fit.llf,
                }
            ]
        ),
        "irls_vs_statsmodels",
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(np.arange(1, ir.n_iter_ + 1), ir.loglik_path_, "o-")
    ax.set_xlabel("Newton-Raphson iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title(rep.title("IRLS convergence (responder model, full-rank design)"))
    rep.figure(fig, "irls_convergence")
    return {"irls_max_abs_coef_diff": diff, "irls_iterations": float(ir.n_iter_)}


# -------------------------------------------------------------------- counts
def _counts(sd: StageData, rep: tracking.Reporter) -> pd.DataFrame:
    cfg = sd.cfg
    tr, te = _rows(sd, cfg["supervised"]["count_population"], None)
    y = "Y_n_inj_yr1"
    cv = predefined_cv(tr, sd.splits)
    pre = outcome_preprocessor(cfg, tr)
    base = ArmMeanRegressor()
    cvr = cross_validate(base, tr, tr[y], cv=cv, scoring=CNT_CV)
    base.fit(tr, tr[y])
    preds = {"Baseline (arm mean)": base.predict(te)}
    rows = [
        {
            "Model": "Baseline (arm mean)",
            **models.summarize_cv(cvr, CNT_KEYS),
            **{
                f"Test_{k}": v
                for k, v in models.count_metrics(te[y], preds["Baseline (arm mean)"]).items()
            },
            "best_params": "",
        }
    ]
    for name, (est, grid) in models.count_models(cfg).items():
        f = models.tune(
            name, est, grid, pre, tr, tr[y].to_numpy(), cv, "neg_mean_poisson_deviance", CNT_CV
        )
        preds[name] = f.model.predict(te)
        row = {
            "Model": name,
            **models.summarize_cv(f.cv_results, CNT_KEYS),
            **{f"Test_{k}": v for k, v in models.count_metrics(te[y], preds[name]).items()},
            "best_params": str(f.best_params),
        }
        rows.append(row)
        tracking.child_run(
            f"count/{name}", f.best_params, {k: v for k, v in row.items() if isinstance(v, float)}
        )
    table = pd.DataFrame(rows)
    rep.table(table, "count_injections_yr1")
    by_arm = pd.DataFrame({"arm": te.treatment, "observed": te[y], **preds}).groupby("arm").mean()
    rep.table(by_arm.reset_index(), "count_by_arm_test")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(by_arm))
    cols = list(by_arm.columns)
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, by_arm[c], w, label=c)
    ax.set_xticks(x, by_arm.index)
    ax.set_ylabel("mean year-1 injections (test patients)")
    ax.legend(fontsize=7)
    ax.set_title(rep.title("Injection-count models: observed vs predicted means by arm"))
    rep.figure(fig, "count_by_arm")
    return table


def run(synthetic: bool) -> None:
    """Run M5 and log to MLflow experiment ``equipoise-m5-supervised`` (nested model runs)."""
    sd = prepare(synthetic)
    rep = tracking.Reporter(MODULE, synthetic, sd.cfg)
    with tracking.start_run(EXPERIMENT, synthetic=synthetic, data_path=sd.data_path, cfg=sd.cfg):
        tracking.log_params({"supervised": sd.cfg["supervised"]})
        reg, info = _regression(sd, rep)
        metrics: dict[str, Any] = {}
        metrics |= _irls_check(sd, rep)
        clf = [_classification(sd, rep, t) for t in ("gain_ge_15", "laser_yr1")]
        _counts(sd, rep)
        summary = pd.concat(
            [c[["target", "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]] for c in clf],
            ignore_index=True,
        )
        rep.table(summary, "classification_summary")
        b = reg.set_index("Model")
        metrics |= {
            "baseline_test_rmse": float(b.loc["Baseline (arm mean)", "Test_RMSE"]),
            "best_cv_rmse": float(b["CV_RMSE"].min()),
        }
        tracking.log_metrics(metrics)
        tracking.log_params(info)
    print(
        f"[supervised] wrote {len(rep.written)} files; best regression by CV: "
        f"{models.REG_LABELS[info['best_regression_model']]}"
    )

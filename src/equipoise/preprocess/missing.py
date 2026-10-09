"""Missing data: complete-case vs KNN imputation vs MICE with Rubin's rules.

44 of 660 patients have no week-52 vision. We estimate the primary arm
differences (aflibercept - bevacizumab, ranibizumab - bevacizumab, in letters)
three ways. *Complete case* analyses only observed patients. *KNN* fills each
missing value from the 5 most similar patients (one imputed dataset). *MICE*
(multiple imputation by chained equations, ``IterativeImputer`` with posterior
sampling) creates M imputed datasets, estimates the difference in each, and
pools them with Rubin's rules, so the confidence interval includes the extra
uncertainty caused by not knowing the missing values. Imputation is done
within each arm so arm-specific relationships are preserved. This is a
trial-level inference on all 660 patients (like the published analysis), not
a model preprocessing step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Pooled:
    """Rubin's-rules pooled estimate."""

    estimate: float
    se: float
    df: float
    ci_low: float
    ci_high: float
    within_var: float
    between_var: float
    fraction_missing_info: float


def rubin_pool(estimates: np.ndarray, variances: np.ndarray, alpha: float = 0.05) -> Pooled:
    """Pool M per-imputation estimates and their variances with Rubin's rules.

    ``Q = mean(q_m)``; within-imputation variance ``W = mean(u_m)``;
    between-imputation variance ``B = var(q_m)``; total ``T = W + (1 + 1/M) B``.
    Degrees of freedom follow Rubin (1987): ``(M-1) (1 + W / ((1 + 1/M) B))^2``.
    """
    q, u = np.asarray(estimates, float), np.asarray(variances, float)
    m = len(q)
    if m < 2:
        raise ValueError("Rubin's rules need at least 2 imputations")
    qbar, w, b = q.mean(), u.mean(), q.var(ddof=1)
    t = w + (1 + 1 / m) * b
    if b > 0:
        r = (1 + 1 / m) * b / w
        dof = (m - 1) * (1 + 1 / r) ** 2
    else:
        dof = np.inf
    crit = stats.t.ppf(1 - alpha / 2, dof) if np.isfinite(dof) else stats.norm.ppf(1 - alpha / 2)
    se = float(np.sqrt(t))
    fmi = float((1 + 1 / m) * b / t) if t > 0 else 0.0
    return Pooled(float(qbar), se, float(dof), qbar - crit * se, qbar + crit * se, w, b, fmi)


def diff_in_means(y: np.ndarray, arm: np.ndarray, a: str, ref: str) -> tuple[float, float]:
    """Difference in means ``a - ref`` and its Welch variance."""
    ya, yr = y[arm == a], y[arm == ref]
    ya, yr = ya[~np.isnan(ya)], yr[~np.isnan(yr)]
    return float(ya.mean() - yr.mean()), float(ya.var(ddof=1) / len(ya) + yr.var(ddof=1) / len(yr))


def _welch_ci(
    est: float, var: float, y: np.ndarray, arm: np.ndarray, a: str, ref: str, alpha: float
) -> tuple[float, float]:
    na, nr = int(np.sum(~np.isnan(y[arm == a]))), int(np.sum(~np.isnan(y[arm == ref])))
    va = np.nanvar(y[arm == a], ddof=1) / na
    vr = np.nanvar(y[arm == ref], ddof=1) / nr
    dof = (va + vr) ** 2 / (va**2 / (na - 1) + vr**2 / (nr - 1))
    crit = stats.t.ppf(1 - alpha / 2, dof)
    return est - crit * np.sqrt(var), est + crit * np.sqrt(var)


def imputation_matrix(df: pd.DataFrame, outcome: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Numeric matrix of baseline features + the outcome for the imputation models."""
    f = cfg["features"]
    cols = list(f["continuous"]) + [f["cst"]["value"]] + list(f["binary"])
    X = df[cols].astype(float).copy()
    for col, order in f["ordinal"].items():
        X[col] = df[col].map({lvl: i for i, lvl in enumerate(order)}).astype(float)
    dummies = pd.get_dummies(df[list(f["nominal"])], dummy_na=False, dtype=float)
    X = pd.concat([X, dummies], axis=1)
    X[outcome] = df[outcome].astype(float)
    return X


def _impute_within_arm(
    X: pd.DataFrame, arm: np.ndarray, make_imputer, within: bool
) -> pd.DataFrame:
    out = X.copy()
    groups = np.unique(arm) if within else [None]
    for g in groups:
        rows = np.ones(len(X), bool) if g is None else arm == g
        sub = X.loc[rows]
        keep = sub.columns[sub.notna().any()]  # drop columns entirely missing in this arm
        scaler = StandardScaler().fit(sub[keep])
        Z = scaler.transform(sub[keep])
        Zi = make_imputer().fit_transform(Z)
        out.loc[rows, keep] = scaler.inverse_transform(Zi)
    return out


def compare_strategies(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arm differences in the primary outcome under the three missing-data strategies.

    Returns ``(summary, per_imputation)``: one row per strategy x comparison,
    and the MICE per-imputation estimates (to show between-imputation spread).
    """
    outcome = cfg["analysis"]["primary_outcome"]
    ref = cfg["analysis"]["reference_arm"]
    others = [a for a in cfg["analysis"]["arms"] if a != ref]
    mcfg = cfg["preprocess"]["missing"]
    alpha = cfg["stats"]["alpha"]
    arm = df["treatment"].to_numpy()
    X = imputation_matrix(df, outcome, cfg)
    y_obs = X[outcome].to_numpy()
    n_missing = int(np.isnan(y_obs).sum())
    rows, per_imp = [], []

    for a in others:
        est, var = diff_in_means(y_obs, arm, a, ref)
        lo, hi = _welch_ci(est, var, y_obs, arm, a, ref, alpha)
        rows.append(
            {
                "strategy": "complete-case",
                "comparison": f"{a} - {ref}",
                "estimate": est,
                "se": np.sqrt(var),
                "ci_low": lo,
                "ci_high": hi,
                "n_used": int(np.sum(~np.isnan(y_obs) & np.isin(arm, [a, ref]))),
            }
        )

    knn = _impute_within_arm(
        X, arm, lambda: KNNImputer(n_neighbors=mcfg["knn_neighbors"]), mcfg["impute_within_arm"]
    )
    y_knn = knn[outcome].to_numpy()
    for a in others:
        est, var = diff_in_means(y_knn, arm, a, ref)
        lo, hi = _welch_ci(est, var, y_knn, arm, a, ref, alpha)
        rows.append(
            {
                "strategy": "KNN (single imputation)",
                "comparison": f"{a} - {ref}",
                "estimate": est,
                "se": np.sqrt(var),
                "ci_low": lo,
                "ci_high": hi,
                "n_used": int(np.isin(arm, [a, ref]).sum()),
            }
        )

    est_m = {a: [] for a in others}
    var_m = {a: [] for a in others}
    for m in range(mcfg["n_imputations"]):
        imp = _impute_within_arm(
            X,
            arm,
            lambda m=m: IterativeImputer(
                sample_posterior=True, max_iter=mcfg["mice_max_iter"], random_state=cfg["seed"] + m
            ),
            mcfg["impute_within_arm"],
        )
        y_m = imp[outcome].to_numpy()
        for a in others:
            e, v = diff_in_means(y_m, arm, a, ref)
            est_m[a].append(e)
            var_m[a].append(v)
            per_imp.append({"imputation": m + 1, "comparison": f"{a} - {ref}", "estimate": e})
    for a in others:
        p = rubin_pool(np.array(est_m[a]), np.array(var_m[a]), alpha)
        rows.append(
            {
                "strategy": f"MICE (M={mcfg['n_imputations']}, Rubin)",
                "comparison": f"{a} - {ref}",
                "estimate": p.estimate,
                "se": p.se,
                "ci_low": p.ci_low,
                "ci_high": p.ci_high,
                "n_used": int(np.isin(arm, [a, ref]).sum()),
                "fraction_missing_info": p.fraction_missing_info,
            }
        )
    summary = pd.DataFrame(rows)
    summary.insert(0, "outcome", outcome)
    summary["n_missing_outcome_total"] = n_missing
    return summary, pd.DataFrame(per_imp)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage missing for every column with any missing value, overall and by arm."""
    cols = [c for c in df.columns if df[c].isna().any()]
    out = pd.DataFrame({"column": cols, "n_missing": [int(df[c].isna().sum()) for c in cols]})
    out["pct_missing"] = 100 * out["n_missing"] / len(df)
    for arm, g in df.groupby("treatment"):
        out[f"n_missing_{arm}"] = [int(g[c].isna().sum()) for c in cols]
    return out.sort_values("n_missing", ascending=False).reset_index(drop=True)

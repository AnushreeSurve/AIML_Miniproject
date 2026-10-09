"""Further inference: variance tests, partial / multiple correlation, interaction regression.

*Levene* (median-centred, i.e. Brown-Forsythe) and *Bartlett* test whether
the arms have equal outcome variances, and pairwise *F-tests* compare each
arm's variance with bevacizumab's. The *partial correlation* between vision
gain and baseline vision controlling for CST is the correlation of the two
residuals after regressing each on CST. The *multiple correlation* R is the
correlation between the gain and its best linear prediction from all
baseline features (R = sqrt(R^2)). Finally the multiple regression
``gain ~ treatment * baseline vision + covariates`` tests heterogeneity: a
non-zero treatment x baseline-vision interaction means the drug effect
depends on baseline vision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm


def variance_tests(df: pd.DataFrame, col: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Levene, Bartlett and pairwise F-tests (arm vs reference) for equal variances."""
    arms, ref = cfg["analysis"]["arms"], cfg["analysis"]["reference_arm"]
    g = {a: df.loc[df.treatment == a, col].dropna().to_numpy() for a in arms}
    lev = stats.levene(*g.values(), center="median")
    bart = stats.bartlett(*g.values())
    rows = [
        {
            "outcome": col,
            "test": "Levene (Brown-Forsythe)",
            "comparison": "all arms",
            "statistic": lev.statistic,
            "p_value": lev.pvalue,
        },
        {
            "outcome": col,
            "test": "Bartlett",
            "comparison": "all arms",
            "statistic": bart.statistic,
            "p_value": bart.pvalue,
        },
    ]
    for a in arms:
        if a == ref:
            continue
        va, vr = np.var(g[a], ddof=1), np.var(g[ref], ddof=1)
        F = va / vr
        d1, d2 = len(g[a]) - 1, len(g[ref]) - 1
        p = 2 * min(stats.f.cdf(F, d1, d2), stats.f.sf(F, d1, d2))
        rows.append(
            {
                "outcome": col,
                "test": "F-test (variance ratio)",
                "comparison": f"{a} / {ref}",
                "statistic": F,
                "p_value": p,
                "sd_arm": np.sqrt(va),
                "sd_ref": np.sqrt(vr),
            }
        )
    return pd.DataFrame(rows)


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """Partial correlation r(x, y | z) by the residual method, with a t-test (df = n - 2 - k)."""
    Z = np.column_stack([np.ones(len(x)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    k = Z.shape[1] - 1
    dof = len(x) - 2 - k
    t = r * np.sqrt(dof / (1 - r**2))
    return {
        "r_partial": r,
        "t": t,
        "df": dof,
        "p_value": 2 * stats.t.sf(abs(t), dof),
        "r_zero_order": float(np.corrcoef(x, y)[0, 1]),
        "n": len(x),
    }


def partial_corr_from_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute the partial correlation from the inverse correlation matrix (a check)."""
    P = np.linalg.inv(np.corrcoef(np.column_stack([x, y, z]), rowvar=False))
    return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))


def multiple_correlation(df: pd.DataFrame, outcome: str, features: list[str]) -> dict[str, float]:
    """Multiple correlation R of ``outcome`` on ``features`` (complete cases, OLS)."""
    import statsmodels.api as sm

    d = df[[outcome, *features]].dropna()
    fit = sm.OLS(d[outcome], sm.add_constant(d[features].astype(float))).fit()
    r_pred = float(np.corrcoef(fit.fittedvalues, d[outcome])[0, 1])
    return {
        "R": float(np.sqrt(fit.rsquared)),
        "R_check_corr_pred": r_pred,
        "R2": fit.rsquared,
        "R2_adj": fit.rsquared_adj,
        "F": fit.fvalue,
        "p_value": fit.f_pvalue,
        "n": int(fit.nobs),
        "k": len(features),
    }


def _q(name: str) -> str:
    return f"Q('{name}')"


def interaction_regression(
    df: pd.DataFrame, cfg: dict[str, Any], moderator: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLS ``outcome ~ treatment * moderator + covariates`` and the interaction F-test."""
    outcome = cfg["analysis"]["primary_outcome"]
    ref = cfg["analysis"]["reference_arm"]
    covs = [c for c in cfg["stats"]["regression_covariates"] if c != moderator]
    d = df[[outcome, "treatment", moderator, *covs]].dropna()
    trt = f"C(treatment, Treatment(reference='{ref}'))"
    rhs_cov = " + ".join(_q(c) for c in covs)
    full = smf.ols(f"{_q(outcome)} ~ {trt} * {_q(moderator)} + {rhs_cov}", data=d).fit()
    reduced = smf.ols(f"{_q(outcome)} ~ {trt} + {_q(moderator)} + {rhs_cov}", data=d).fit()
    ci = full.conf_int(cfg["stats"]["alpha"])
    coefs = pd.DataFrame(
        {
            "model": f"treatment x {moderator}",
            "term": full.params.index,
            "coef": full.params.to_numpy(),
            "se": full.bse.to_numpy(),
            "ci_low": ci[0].to_numpy(),
            "ci_high": ci[1].to_numpy(),
            "p_value": full.pvalues.to_numpy(),
        }
    )
    coefs["term"] = (
        coefs["term"]
        .str.replace(f"{trt}", "treatment", regex=False)
        .str.replace("Q('", "", regex=False)
        .str.replace("')", "", regex=False)
    )
    a = anova_lm(reduced, full)
    test = pd.DataFrame(
        [
            {
                "model": f"treatment x {moderator}",
                "n": int(full.nobs),
                "F_interaction": float(a["F"].iloc[1]),
                "df_num": float(a["df_diff"].iloc[1]),
                "p_interaction": float(a["Pr(>F)"].iloc[1]),
                "R2_full": full.rsquared,
                "R2_reduced": reduced.rsquared,
            }
        ]
    )
    return coefs, test

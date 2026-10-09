"""Replication of the Protocol T results at 1 and 2 years.

For vision we report the mean letters gained per arm with a 95 % t-interval,
a one-way ANOVA across the three arms and pairwise Welch t-tests (which do
not assume equal variances) with Holm correction, overall and within the
published baseline-vision subgroups. Injection counts are skewed, so we
report medians and compare arms with the Mann-Whitney U test; laser rates
are proportions compared with chi-square. Denominators follow README 9.1:
year-1 injections and laser use patients with a week-52 visit, 2-year ones
patients with a week-104 visit; vision uses every observed value (complete
cases).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from equipoise.stats.multiple import holm


def mean_ci(x: pd.Series, alpha: float) -> dict[str, float]:
    """n, mean, SD and t-based CI of the mean."""
    x = x.dropna()
    n, m, s = len(x), float(x.mean()), float(x.std(ddof=1))
    half = stats.t.ppf(1 - alpha / 2, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "mean": m, "sd": s, "ci_low": m - half, "ci_high": m + half}


def _pairs(arms: list[str], ref: str) -> list[tuple[str, str]]:
    """All arm pairs, with comparisons against ``ref`` first and ``ref`` second in each."""
    against = [(a, ref) for a in arms if a != ref]
    rest = [p for p in combinations([a for a in arms if a != ref], 2)]
    return against + rest


def continuous_outcome(
    df: pd.DataFrame, col: str, cfg: dict[str, Any], label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-arm means with CIs, plus ANOVA and Holm-corrected pairwise Welch t-tests."""
    arms, ref, alpha = (
        cfg["analysis"]["arms"],
        cfg["analysis"]["reference_arm"],
        cfg["stats"]["alpha"],
    )
    means = pd.DataFrame(
        [
            {
                "analysis": label,
                "outcome": col,
                "arm": a,
                **mean_ci(df.loc[df.treatment == a, col], alpha),
            }
            for a in arms
        ]
    )
    groups = [df.loc[df.treatment == a, col].dropna() for a in arms]
    f = stats.f_oneway(*groups)
    tests = [
        {
            "analysis": label,
            "outcome": col,
            "comparison": "all arms",
            "test": "one-way ANOVA",
            "statistic": f.statistic,
            "p_value": f.pvalue,
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    ]
    pair_rows = []
    for a, b in _pairs(arms, ref):
        xa, xb = (df.loc[df.treatment == a, col].dropna(), df.loc[df.treatment == b, col].dropna())
        t = stats.ttest_ind(xa, xb, equal_var=False)
        ci = t.confidence_interval(1 - alpha)
        pair_rows.append(
            {
                "analysis": label,
                "outcome": col,
                "comparison": f"{a} - {b}",
                "test": "Welch t",
                "statistic": t.statistic,
                "p_value": t.pvalue,
                "estimate": float(xa.mean() - xb.mean()),
                "ci_low": ci.low,
                "ci_high": ci.high,
            }
        )
    adj = holm([r["p_value"] for r in pair_rows])
    for r, p in zip(pair_rows, adj, strict=True):
        r["p_holm"] = p
    return means, pd.DataFrame(tests + pair_rows)


def count_outcome(
    df: pd.DataFrame, col: str, cfg: dict[str, Any], label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-arm median [IQR], Kruskal-Wallis and Holm-corrected pairwise Mann-Whitney U."""
    arms, ref = cfg["analysis"]["arms"], cfg["analysis"]["reference_arm"]
    rows = []
    for a in arms:
        x = df.loc[df.treatment == a, col].dropna()
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "analysis": label,
                "outcome": col,
                "arm": a,
                "n": len(x),
                "median": med,
                "q1": q1,
                "q3": q3,
                "mean": x.mean(),
            }
        )
    groups = [df.loc[df.treatment == a, col].dropna() for a in arms]
    kw = stats.kruskal(*groups)
    tests = [
        {
            "analysis": label,
            "outcome": col,
            "comparison": "all arms",
            "test": "Kruskal-Wallis",
            "statistic": kw.statistic,
            "p_value": kw.pvalue,
        }
    ]
    pair_rows = []
    for a, b in _pairs(arms, ref):
        u = stats.mannwhitneyu(
            df.loc[df.treatment == a, col].dropna(),
            df.loc[df.treatment == b, col].dropna(),
            alternative="two-sided",
        )
        pair_rows.append(
            {
                "analysis": label,
                "outcome": col,
                "comparison": f"{a} vs {b}",
                "test": "Mann-Whitney U",
                "statistic": u.statistic,
                "p_value": u.pvalue,
            }
        )
    for r, p in zip(pair_rows, holm([r["p_value"] for r in pair_rows]), strict=True):
        r["p_holm"] = p
    return pd.DataFrame(rows), pd.DataFrame(tests + pair_rows)


def binary_outcome(
    df: pd.DataFrame, col: str, cfg: dict[str, Any], label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-arm rate with Wilson CI, chi-square overall and Holm-corrected pairwise chi-square."""
    from statsmodels.stats.proportion import proportion_confint

    arms, ref, alpha = (
        cfg["analysis"]["arms"],
        cfg["analysis"]["reference_arm"],
        cfg["stats"]["alpha"],
    )
    rows = []
    for a in arms:
        x = df.loc[df.treatment == a, col].dropna()
        k, n = int(x.sum()), len(x)
        lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
        rows.append(
            {
                "analysis": label,
                "outcome": col,
                "arm": a,
                "n": n,
                "events": k,
                "pct": 100 * k / n,
                "ci_low_pct": 100 * lo,
                "ci_high_pct": 100 * hi,
            }
        )
    tab = pd.crosstab(df["treatment"], df[col]).reindex(arms)
    chi2, p, _, _ = stats.chi2_contingency(tab.to_numpy())
    tests = [
        {
            "analysis": label,
            "outcome": col,
            "comparison": "all arms",
            "test": "chi-square",
            "statistic": chi2,
            "p_value": p,
        }
    ]
    pair_rows = []
    for a, b in _pairs(arms, ref):
        c2, pp, _, _ = stats.chi2_contingency(tab.loc[[a, b]].to_numpy())
        pair_rows.append(
            {
                "analysis": label,
                "outcome": col,
                "comparison": f"{a} vs {b}",
                "test": "chi-square",
                "statistic": c2,
                "p_value": pp,
            }
        )
    for r, q in zip(pair_rows, holm([r["p_value"] for r in pair_rows]), strict=True):
        r["p_holm"] = q
    return pd.DataFrame(rows), pd.DataFrame(tests + pair_rows)


def replicate(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run every replication analysis; return named result tables."""
    s = cfg["stats"]
    yr1 = df[df[s["yr1_population"]] == 1]
    yr2 = df[df[s["yr2_population"]] == 1]
    cut = cfg["trial"]["va_below_69_cutoff"]
    means, tests = [], []
    for data, col, label in [
        (df, "Y_va_change_wk52", "week 52, all"),
        (
            df[df.X_va_below_69 == 1],
            "Y_va_change_wk52",
            f"week 52, baseline <= {cut} letters (20/50 or worse)",
        ),
        (
            df[df.X_va_below_69 == 0],
            "Y_va_change_wk52",
            f"week 52, baseline >= {cut + 1} letters (20/32-20/40)",
        ),
        (df, "Y_va_change_wk104", "week 104, all"),
        (df, "Y_cst_change_wk52", "week 52, all"),
        (df, "Y_cst_change_wk104", "week 104, all"),
    ]:
        m, t = continuous_outcome(data, col, cfg, label)
        means.append(m)
        tests.append(t)
    counts, ctests = [], []
    for data, col, label in [
        (yr1, "Y_n_inj_yr1", "year 1 (week-52 visit)"),
        (yr2, "Y_n_inj_total", "2 years (week-104 visit)"),
    ]:
        m, t = count_outcome(data, col, cfg, label)
        counts.append(m)
        ctests.append(t)
    rates, rtests = [], []
    for data, col, label in [
        (yr1, "Y_laser_yr1", "year 1 (week-52 visit)"),
        (yr2, "Y_laser_any", "2 years (week-104 visit)"),
    ]:
        m, t = binary_outcome(data, col, cfg, label)
        rates.append(m)
        rtests.append(t)
    return {
        "means": pd.concat(means, ignore_index=True),
        "mean_tests": pd.concat(tests, ignore_index=True),
        "injections": pd.concat(counts, ignore_index=True),
        "injection_tests": pd.concat(ctests, ignore_index=True),
        "laser": pd.concat(rates, ignore_index=True),
        "laser_tests": pd.concat(rtests, ignore_index=True),
    }


def replication_check(res: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Compare our numbers with the README 9.1 targets (config ``stats.replication_targets``)."""
    tg = cfg["stats"]["replication_targets"]
    tol = cfg["stats"]["replication_tolerance_letters"]
    m, inj, las = res["means"], res["injections"], res["laser"]

    def mean_of(outcome, label_start, arm):
        r = m[(m.outcome == outcome) & m.analysis.str.startswith(label_start) & (m.arm == arm)]
        return float(r["mean"].iloc[0])

    getters = {
        "mean_gain_wk52": (lambda a: mean_of("Y_va_change_wk52", "week 52, all", a), "abs", tol),
        "mean_gain_wk52_va_below_69": (
            lambda a: mean_of("Y_va_change_wk52", "week 52, baseline <=", a),
            "abs",
            tol,
        ),
        "mean_gain_wk52_va_69_plus": (
            lambda a: mean_of("Y_va_change_wk52", "week 52, baseline >=", a),
            "abs",
            tol,
        ),
        "mean_gain_wk104": (lambda a: mean_of("Y_va_change_wk104", "week 104", a), "abs", tol),
        "cst_change_wk52": (lambda a: mean_of("Y_cst_change_wk52", "week 52", a), "round", 0),
        "median_inj_yr1": (
            lambda a: float(inj[(inj.outcome == "Y_n_inj_yr1") & (inj.arm == a)]["median"].iloc[0]),
            "round",
            0,
        ),
        "median_inj_total": (
            lambda a: float(
                inj[(inj.outcome == "Y_n_inj_total") & (inj.arm == a)]["median"].iloc[0]
            ),
            "round",
            0,
        ),
        "laser_yr1_pct": (
            lambda a: float(las[(las.outcome == "Y_laser_yr1") & (las.arm == a)]["pct"].iloc[0]),
            "round",
            0,
        ),
        "laser_any_pct": (
            lambda a: float(las[(las.outcome == "Y_laser_any") & (las.arm == a)]["pct"].iloc[0]),
            "round",
            0,
        ),
    }
    rows = []
    for metric, targets in tg.items():
        get, mode, t = getters[metric]
        for arm, target in targets.items():
            ours = get(arm)
            ok = abs(ours - target) <= t + 1e-9 if mode == "abs" else round(ours) == round(target)
            rows.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "readme_target": target,
                    "ours": ours,
                    "rule": f"|diff| <= {t}" if mode == "abs" else "equal after rounding",
                    "match": bool(ok),
                }
            )
    return pd.DataFrame(rows)

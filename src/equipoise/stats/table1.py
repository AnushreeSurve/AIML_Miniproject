"""Table 1: baseline characteristics by randomized arm.

Shows that randomization balanced the arms. Continuous features are
summarized as mean +/- SD, or median [IQR] when skewed (|skewness| above the
config threshold), and compared with one-way ANOVA or Kruskal-Wallis
respectively. Binary and categorical features are shown as n (%) and
compared with a chi-square test. In a randomized trial these p-values should
mostly be large; a few small ones are expected by chance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from equipoise.data.schema import ANALYSIS_SPEC


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _chi2(table: pd.DataFrame) -> tuple[float, str]:
    table = table.loc[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        return np.nan, "chi-square"
    chi2, p, _, _ = stats.chi2_contingency(table.to_numpy())
    return p, "chi-square"


def table1(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build Table 1 (one row per feature or category level)."""
    arms = cfg["analysis"]["arms"]
    f = cfg["features"]
    cont = list(f["continuous"]) + [f["cst"]["value"]]
    binary = list(f["binary"])
    cats = list(f["nominal"]) + list(f["ordinal"])
    thr = cfg["stats"]["skew_threshold"]
    n = df["treatment"].value_counts()
    rows: list[dict[str, Any]] = []

    for c in cont:
        x = df[c]
        skewed = abs(float(x.skew())) > thr
        groups = [df.loc[df.treatment == a, c].dropna() for a in arms]
        row = {
            "characteristic": ANALYSIS_SPEC[c].description,
            "feature": c,
            "summary": "median [IQR]" if skewed else "mean +/- SD",
        }
        for a, g in zip(arms, groups, strict=True):
            if skewed:
                q1, med, q3 = g.quantile([0.25, 0.5, 0.75])
                row[a] = f"{med:.1f} [{q1:.1f}, {q3:.1f}]"
            else:
                row[a] = f"{g.mean():.1f} +/- {g.std():.1f}"
        test = stats.kruskal(*groups) if skewed else stats.f_oneway(*groups)
        row |= {
            "test": "Kruskal-Wallis" if skewed else "ANOVA",
            "p_value": test.pvalue,
            "n_missing": int(x.isna().sum()),
        }
        rows.append(row)

    for c in binary:
        tab = pd.crosstab(df[c], df["treatment"]).reindex(columns=arms, fill_value=0)
        p, test = _chi2(tab)
        row = {"characteristic": ANALYSIS_SPEC[c].description, "feature": c, "summary": "n (%)"}
        for a in arms:
            k = int(tab.loc[1, a]) if 1 in tab.index else 0
            denom = int(df.loc[df.treatment == a, c].notna().sum())
            row[a] = f"{k} ({100 * k / denom:.1f}%)"
        row |= {"test": test, "p_value": p, "n_missing": int(df[c].isna().sum())}
        rows.append(row)

    for c in cats:
        levels = f["ordinal"].get(c) or ANALYSIS_SPEC[c].categories
        tab = pd.crosstab(df[c], df["treatment"]).reindex(index=levels, columns=arms, fill_value=0)
        p, test = _chi2(tab)
        rows.append(
            {
                "characteristic": ANALYSIS_SPEC[c].description,
                "feature": c,
                "summary": "n (%)",
                **dict.fromkeys(arms, ""),
                "test": test,
                "p_value": p,
                "n_missing": int(df[c].isna().sum()),
            }
        )
        for lvl in levels:
            row = {"characteristic": f"  {lvl}", "feature": c, "summary": ""}
            for a in arms:
                denom = int(df.loc[df.treatment == a, c].notna().sum())
                k = int(tab.loc[lvl, a])
                row[a] = f"{k} ({100 * k / denom:.1f}%)"
            rows.append(row | {"test": "", "p_value": np.nan, "n_missing": np.nan})

    out = pd.DataFrame(rows)
    header = {a: f"{a} (n={int(n.get(a, 0))})" for a in arms}
    out["p"] = out["p_value"].map(lambda v: "" if pd.isna(v) else _fmt_p(v))
    return out.rename(columns=header)

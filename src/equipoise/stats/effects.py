"""Arm differences in mean outcome (Welch t-interval), reused by exploratory analyses."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def arm_differences(
    df: pd.DataFrame, outcome: str, cfg: dict[str, Any], min_n: int = 2
) -> list[dict[str, Any]]:
    """Each arm minus the reference arm: estimate, Welch 95 % CI and group sizes.

    Comparisons where either arm has fewer than ``min_n`` observed outcomes
    are returned with empty estimates rather than an unstable number.
    """
    ref, alpha = cfg["analysis"]["reference_arm"], cfg["stats"]["alpha"]
    yr = df.loc[df.treatment == ref, outcome].dropna()
    rows = []
    for a in cfg["analysis"]["arms"]:
        if a == ref:
            continue
        ya = df.loc[df.treatment == a, outcome].dropna()
        row = {
            "comparison": f"{a} - {ref}",
            "n_arm": len(ya),
            "n_ref": len(yr),
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
        if len(ya) >= min_n and len(yr) >= min_n:
            t = stats.ttest_ind(ya, yr, equal_var=False)
            ci = t.confidence_interval(1 - alpha)
            row |= {
                "estimate": float(ya.mean() - yr.mean()),
                "ci_low": ci.low,
                "ci_high": ci.high,
                "p_value": t.pvalue,
            }
        rows.append(row)
    return rows

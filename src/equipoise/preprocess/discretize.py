"""Discretization of baseline features into clinically readable bands.

Association-rule mining (M3) needs categorical "items", so continuous
features are cut into bands at fixed, clinically meaningful edges from
config: HbA1c bands, vision bands on the Snellen scale, age bands, and the
harmonized CST z-score. Fixed edges (rather than quantiles learned from the
data) mean the bands never depend on which patients are in the training set.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from equipoise.config import load_config


def discretize(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return one categorical column ``<feature>_band`` per configured feature."""
    cfg = cfg or load_config()
    out = pd.DataFrame(index=df.index)
    for col, spec in cfg["preprocess"]["bands"].items():
        if col not in df.columns:
            continue
        out[f"{col}_band"] = pd.cut(
            df[col], bins=spec["edges"], labels=spec["labels"], right=False, include_lowest=True
        )
    return out

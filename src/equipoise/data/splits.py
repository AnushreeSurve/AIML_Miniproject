"""Train/test split.

One fixed held-out test set (20 % by default) is drawn once, stratified by
randomized arm, with the seed from ``config/default.yaml``. It is touched only
for final reporting; all model selection happens by cross-validation inside
the remaining 80 %. Splitting is done on patient IDs so the same split can be
saved and re-applied to every table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from equipoise.config import load_config


def make_test_split(
    df: pd.DataFrame,
    *,
    test_fraction: float | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Return sorted ``(train_ids, test_ids)`` patient IDs, stratified by arm."""
    cfg = cfg or load_config()
    test_fraction = cfg["split"]["test_fraction"] if test_fraction is None else test_fraction
    seed = cfg["seed"] if seed is None else seed
    stratify_by = stratify_by or cfg["split"]["stratify_by"]
    train_ids, test_ids = train_test_split(
        df["pt_id"].to_numpy(),
        test_size=test_fraction,
        random_state=seed,
        stratify=df[stratify_by].to_numpy(),
    )
    return sorted(int(i) for i in train_ids), sorted(int(i) for i in test_ids)

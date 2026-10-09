"""Leakage guard for model inputs.

Only baseline ``X_`` columns may be model inputs. ``Y_`` columns are targets,
``M_`` columns are bookkeeping, and ``treatment`` is allowed only when a
learner explicitly takes it as the treatment variable. ``assert_no_leakage``
enforces this as a whitelist, and also understands sklearn
``ColumnTransformer`` output names such as ``num__X_age`` or
``cat__X_race_White``.

Every function that builds a baseline feature matrix must be registered with
``@register_feature_builder`` so ``tests/test_leakage.py`` checks it
automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

TREATMENT_COL = "treatment"
FEATURE_PREFIX = "X_"

#: name -> callable(df) returning a feature matrix (DataFrame or ndarray with names).
BASELINE_FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


class LeakageError(AssertionError):
    """A non-baseline column reached a model's feature matrix."""


def _base_name(name: str) -> str:
    """Strip a ColumnTransformer prefix (``num__X_age`` -> ``X_age``)."""
    return name.rsplit("__", 1)[-1]


def assert_no_leakage(columns: Iterable[str], *, allow_treatment: bool = False) -> None:
    """Raise ``LeakageError`` unless every column is a baseline ``X_`` feature.

    ``allow_treatment=True`` additionally permits ``treatment`` (or its
    one-hot columns ``treatment_<arm>``) for learners that take the arm as an
    explicit input.
    """
    bad = []
    for col in columns:
        base = _base_name(str(col))
        if base.startswith(FEATURE_PREFIX):
            continue
        if allow_treatment and (base == TREATMENT_COL or base.startswith(f"{TREATMENT_COL}_")):
            continue
        bad.append(col)
    if bad:
        raise LeakageError(
            f"non-baseline columns in a feature matrix: {bad}. Only X_ columns may be "
            "model inputs (treatment only where a learner takes it explicitly)."
        )


def register_feature_builder(name: str):
    """Register a baseline feature-matrix builder for the leakage test."""

    def deco(fn: Callable[[pd.DataFrame], pd.DataFrame]):
        BASELINE_FEATURE_BUILDERS[name] = fn
        return fn

    return deco


@register_feature_builder("baseline_features")
def baseline_features(df: pd.DataFrame, *, with_treatment: bool = False) -> pd.DataFrame:
    """Select the baseline ``X_`` columns (plus ``treatment`` if requested), checked for leakage."""
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if with_treatment:
        cols.append(TREATMENT_COL)
    out = df[cols]
    assert_no_leakage(out.columns, allow_treatment=with_treatment)
    return out

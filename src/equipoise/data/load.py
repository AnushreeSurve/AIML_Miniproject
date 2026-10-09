"""Load the analysis and visit tables, real or synthetic.

Real data lives under ``data/`` (never committed; built from the DRCR download).
Synthetic data with the identical schema lives under ``data/synthetic/`` and is
used by tests and CI. Every load is validated against ``schema.py``; real data
is additionally checked against the published trial facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from equipoise.config import get_path, load_config, resolve_path
from equipoise.data.schema import (
    ANALYSIS_SPEC,
    VISIT_SPEC,
    coerce_dtypes,
    validate_analysis,
    validate_visits,
)

ANALYSIS_FILE = "protocolT_analysis.csv"
VISITS_FILE = "protocolT_visits.csv"
TRUTH_FILE = "protocolT_truth.csv"


def analysis_path(synthetic: bool, cfg: dict[str, Any] | None = None) -> Path:
    """Path of the analysis CSV for the chosen data source."""
    cfg = cfg or load_config()
    if synthetic:
        return resolve_path(cfg["paths"]["synthetic_dir"]) / ANALYSIS_FILE
    return get_path("analysis_csv", cfg)


def visits_path(synthetic: bool, cfg: dict[str, Any] | None = None) -> Path:
    """Path of the visit-level CSV for the chosen data source."""
    cfg = cfg or load_config()
    if synthetic:
        return resolve_path(cfg["paths"]["synthetic_dir"]) / VISITS_FILE
    return get_path("visits_csv", cfg)


def _require(path: Path, synthetic: bool) -> None:
    if path.exists():
        return
    hint = (
        "Run `python scripts/make_synthetic_data.py`."
        if synthetic
        else "Build it from the DRCR download (README section 6.3), or add --synthetic."
    )
    raise FileNotFoundError(f"{path} not found. {hint}")


def load_analysis(
    synthetic: bool = False, *, cfg: dict[str, Any] | None = None, validate: bool = True
) -> pd.DataFrame:
    """Load and validate the one-row-per-patient table."""
    cfg = cfg or load_config()
    path = analysis_path(synthetic, cfg)
    _require(path, synthetic)
    df = coerce_dtypes(pd.read_csv(path), ANALYSIS_SPEC)
    if validate:
        validate_analysis(df, expect_trial=not synthetic, trial=cfg["trial"])
    return df


def load_visits(
    synthetic: bool = False,
    *,
    cfg: dict[str, Any] | None = None,
    validate: bool = True,
    analysis: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load and validate the patient x visit table (cross-checked against ``analysis``)."""
    cfg = cfg or load_config()
    path = visits_path(synthetic, cfg)
    _require(path, synthetic)
    df = coerce_dtypes(pd.read_csv(path), VISIT_SPEC)
    if validate:
        validate_visits(df, analysis=analysis)
    return df


def load_synthetic_truth(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Planted potential-outcome means and effects for the synthetic patients."""
    cfg = cfg or load_config()
    path = resolve_path(cfg["paths"]["synthetic_dir"]) / TRUTH_FILE
    _require(path, synthetic=True)
    return pd.read_csv(path)

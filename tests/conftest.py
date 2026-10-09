"""Shared fixtures. Tests run on data/synthetic/ so CI needs no DRCR data."""

from __future__ import annotations

import pandas as pd
import pytest

from equipoise.config import get_path, load_config
from equipoise.data.load import load_analysis, load_synthetic_truth, load_visits


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def syn_analysis() -> pd.DataFrame:
    return load_analysis(synthetic=True)


@pytest.fixture(scope="session")
def syn_visits(syn_analysis) -> pd.DataFrame:
    return load_visits(synthetic=True, analysis=syn_analysis)


@pytest.fixture(scope="session")
def syn_truth() -> pd.DataFrame:
    return load_synthetic_truth()


@pytest.fixture(scope="session")
def real_analysis() -> pd.DataFrame:
    if not get_path("analysis_csv").exists():
        pytest.skip("real DRCR analysis table not present")
    return load_analysis(synthetic=False)

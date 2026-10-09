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


@pytest.fixture()
def fast_config(tmp_path, monkeypatch):
    """Config copy with small MICE settings and all outputs redirected to ``tmp_path``."""
    import shutil

    import yaml

    from equipoise import config as config_mod

    cdir = tmp_path / "config"
    shutil.copytree(config_mod.CONFIG_DIR, cdir)
    d = yaml.safe_load((cdir / "default.yaml").read_text())
    d["preprocess"]["missing"]["n_imputations"] = 3
    d["preprocess"]["missing"]["mice_max_iter"] = 3
    d["paths"]["synthetic_reports_dir"] = str(tmp_path / "reports")
    d["paths"]["splits_json"] = str(tmp_path / "splits.json")
    d["tracking"]["mlflow_tracking_uri"] = str(tmp_path / "mlruns")
    (cdir / "default.yaml").write_text(yaml.safe_dump(d))
    monkeypatch.setattr(config_mod, "CONFIG_DIR", cdir)
    config_mod._load_cached.cache_clear()
    yield tmp_path
    config_mod._load_cached.cache_clear()

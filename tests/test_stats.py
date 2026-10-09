import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from equipoise.stats import inference, replication
from equipoise.stats.multiple import holm
from equipoise.stats.table1 import table1


def test_holm_matches_statsmodels():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.001])
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])


def test_partial_correlation_two_ways():
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + rng.normal(size=300)
    y = z - 0.5 * x + rng.normal(size=300)
    r = inference.partial_correlation(x, y, z)["r_partial"]
    assert np.isclose(r, inference.partial_corr_from_matrix(x, y, z))


def test_f_test_two_sided():
    df = pd.DataFrame(
        {
            "treatment": ["Aflibercept"] * 50 + ["Bevacizumab"] * 50 + ["Ranibizumab"] * 50,
            "y": np.r_[np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.linspace(-3, 3, 50)],
        }
    )
    cfg = {
        "analysis": {
            "arms": ["Aflibercept", "Bevacizumab", "Ranibizumab"],
            "reference_arm": "Bevacizumab",
        }
    }
    t = inference.variance_tests(df, "y", cfg).set_index("comparison")
    assert np.isclose(t.loc["Aflibercept / Bevacizumab", "statistic"], 1.0)
    assert t.loc["Aflibercept / Bevacizumab", "p_value"] > 0.99
    assert t.loc["Ranibizumab / Bevacizumab", "p_value"] < 1e-6


def test_table1_has_all_features(syn_analysis, cfg):
    t = table1(syn_analysis, cfg)
    f = cfg["features"]
    expected = (
        set(f["continuous"])
        | {f["cst"]["value"]}
        | set(f["binary"])
        | set(f["nominal"])
        | set(f["ordinal"])
    )
    assert set(t["feature"]) == expected
    assert t["p_value"].dropna().between(0, 1).all()


def test_interaction_regression_detects_planted_effect(syn_analysis, cfg):
    df = syn_analysis.assign(
        X_cst_um_harm=(syn_analysis.X_cst_um - syn_analysis.X_cst_um.mean())
        / syn_analysis.X_cst_um.std()
    )
    _, test = inference.interaction_regression(df, cfg, "X_va_below_69")
    assert test["p_interaction"].iloc[0] < 0.05


def test_replication_check_shape(syn_analysis, cfg):
    res = replication.replicate(syn_analysis, cfg)
    check = replication.replication_check(res, cfg)
    n_targets = sum(len(v) for v in cfg["stats"]["replication_targets"].values())
    assert len(check) == n_targets


def test_real_data_replicates_readme(real_analysis, cfg):
    """README 9.1: complete-case means within 0.2 letters; medians and laser % equal."""
    res = replication.replicate(real_analysis, cfg)
    check = replication.replication_check(res, cfg)
    bad = check[~check["match"]]
    assert bad.empty, bad.to_string()


@pytest.mark.parametrize("stage", ["preprocess", "stats"])
def test_stage_runs_on_synthetic(stage, fast_config):
    import importlib

    importlib.import_module(f"equipoise.{stage}.stage").run(True)
    figs = list((fast_config / "reports" / "figures").glob("*.png"))
    tabs = list((fast_config / "reports" / "tables").glob("*.md"))
    assert figs and tabs
    assert all(p.with_suffix(".csv").exists() for p in tabs)

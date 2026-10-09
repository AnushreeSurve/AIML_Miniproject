"""Leakage guard: no Y_/M_ column or treatment may reach a baseline feature matrix."""

import pytest

from equipoise.data.features import (
    BASELINE_FEATURE_BUILDERS,
    LeakageError,
    assert_no_leakage,
    baseline_features,
    load_builders,
)

load_builders()


@pytest.mark.parametrize("name", sorted(BASELINE_FEATURE_BUILDERS))
def test_registered_builders_are_baseline_only(name, syn_analysis):
    X = BASELINE_FEATURE_BUILDERS[name](syn_analysis)
    cols = list(X.columns) if hasattr(X, "columns") else list(X.feature_names)
    assert cols, f"{name} produced no columns"
    assert_no_leakage(cols)
    assert not any(c.startswith(("Y_", "M_")) or c == "treatment" for c in cols)


def test_baseline_features_selects_all_x(syn_analysis):
    X = baseline_features(syn_analysis)
    assert set(X.columns) == {c for c in syn_analysis.columns if c.startswith("X_")}


@pytest.mark.parametrize(
    "bad",
    [
        "Y_va_change_wk52",
        "M_has_wk52",
        "treatment",
        "pt_id",
        "study_eye",
        "num__Y_n_inj_yr1",
        "cat__M_final_status_Completed",
    ],
)
def test_non_baseline_columns_rejected(bad):
    with pytest.raises(LeakageError):
        assert_no_leakage(["X_age", bad])


def test_treatment_allowed_only_explicitly():
    assert_no_leakage(["X_age", "treatment"], allow_treatment=True)
    assert_no_leakage(["num__X_age", "cat__treatment_Aflibercept"], allow_treatment=True)
    with pytest.raises(LeakageError):
        assert_no_leakage(["X_age", "treatment"])
    with pytest.raises(LeakageError):
        assert_no_leakage(["treatment", "Y_laser_yr1"], allow_treatment=True)


def test_transformer_names_accepted():
    assert_no_leakage(["num__X_age", "cat__X_race_White", "X_va_below_69"])

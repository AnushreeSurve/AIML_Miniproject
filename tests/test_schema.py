import numpy as np
import pytest

from equipoise.data.schema import (
    ANALYSIS_COLUMNS,
    SchemaError,
    column_table,
    validate_analysis,
    validate_visits,
)


def test_synthetic_analysis_matches_schema(syn_analysis, cfg):
    assert list(syn_analysis.columns) == [c.name for c in ANALYSIS_COLUMNS]
    validate_analysis(syn_analysis, trial=cfg["trial"])


def test_synthetic_visits_match_schema(syn_visits, syn_analysis):
    validate_visits(syn_visits, analysis=syn_analysis)


def test_real_table_matches_trial(real_analysis, cfg):
    validate_analysis(real_analysis, expect_trial=True, trial=cfg["trial"])


def test_missing_column_fails(syn_analysis):
    with pytest.raises(SchemaError, match="missing columns"):
        validate_analysis(syn_analysis.drop(columns=["X_age"]))


def test_extra_column_fails(syn_analysis):
    with pytest.raises(SchemaError, match="unexpected columns"):
        validate_analysis(syn_analysis.assign(X_new=1))


def test_out_of_range_fails(syn_analysis):
    bad = syn_analysis.copy()
    bad.loc[0, "X_va_letters"] = 120.0
    with pytest.raises(SchemaError, match="X_va_letters"):
        validate_analysis(bad)


def test_unknown_category_fails(syn_analysis):
    bad = syn_analysis.copy()
    bad.loc[0, "treatment"] = "Placebo"
    with pytest.raises(SchemaError, match="treatment"):
        validate_analysis(bad)


def test_non_nullable_missing_fails(syn_analysis):
    bad = syn_analysis.copy()
    bad["X_sbp"] = bad["X_sbp"].astype(float)
    bad.loc[0, "X_sbp"] = np.nan
    with pytest.raises(SchemaError, match="X_sbp"):
        validate_analysis(bad)


def test_has_wk52_consistency(syn_analysis):
    bad = syn_analysis.copy()
    i = bad.index[bad["M_has_wk52"] == 1][0]
    bad.loc[i, "Y_va_change_wk52"] = np.nan
    with pytest.raises(SchemaError, match="M_has_wk52"):
        validate_analysis(bad)


def test_trial_facts_fail_on_synthetic_subset(syn_analysis, cfg):
    with pytest.raises(SchemaError, match="patients"):
        validate_analysis(syn_analysis.iloc[:100], expect_trial=True, trial=cfg["trial"])


def test_visit_arm_mismatch_fails(syn_visits, syn_analysis):
    bad = syn_visits.copy()
    bad.loc[0, "treatment"] = next(a for a in cfg_arms() if a != bad.loc[0, "treatment"])
    with pytest.raises(SchemaError, match="arm"):
        validate_visits(bad, analysis=syn_analysis)


def test_feature_description_table():
    t = column_table()
    assert list(t.columns) == ["Feature", "Data Type", "Description", "Role"]
    assert (t["Description"].str.len() > 0).all()


def cfg_arms():
    return ("Aflibercept", "Bevacizumab", "Ranibizumab")

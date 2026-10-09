import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from equipoise.data.features import assert_no_leakage
from equipoise.data.splits import load_splits, split_frame
from equipoise.preprocess.encode import transform_frame
from equipoise.supervised import knn_timing, models
from equipoise.supervised.design import (
    ArmMeanRegressor,
    ArmRateClassifier,
    outcome_preprocessor,
    predefined_cv,
)
from equipoise.supervised.scratch import IRLSLogisticRegression, normal_equations


def test_irls_matches_statsmodels():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 6))
    y = (rng.random(500) < 1 / (1 + np.exp(-(X @ np.array([1, -1, 0.5, 0, 0.3, -0.2]))))).astype(
        int
    )
    ours = IRLSLogisticRegression().fit(X, y)
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    assert np.allclose(np.r_[ours.intercept_, ours.coef_.ravel()], ref.params, atol=1e-8)
    assert np.all(np.diff(ours.loglik_path_) >= -1e-9)  # Newton steps never decrease log-lik here


def test_irls_l2_handles_collinear_design():
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    X = np.column_stack([a, a])  # perfectly collinear
    y = (a + rng.normal(size=200) > 0).astype(int)
    m = IRLSLogisticRegression(l2=1e-3).fit(X, y)
    assert np.isclose(m.coef_[0, 0], m.coef_[0, 1])


def test_normal_equations_match_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5))
    y = X @ np.arange(1, 6) + rng.normal(size=300)
    lr = LinearRegression().fit(X, y)
    assert np.allclose(normal_equations(X, y), np.r_[lr.intercept_, lr.coef_])


def test_outcome_design_has_arm_but_no_outcomes(syn_analysis, cfg):
    train, test = split_frame(syn_analysis, load_splits(True, syn_analysis))
    pre = outcome_preprocessor(cfg, train).fit(train)
    Z = transform_frame(pre, test)
    assert_no_leakage(Z.columns, allow_treatment=True)
    arm_cols = [c for c in Z.columns if "treatment_" in c]
    assert sorted(arm_cols) == ["arm__treatment_Aflibercept", "arm__treatment_Ranibizumab"]
    assert not any(c.rsplit("__", 1)[-1].startswith(("Y_", "M_")) for c in Z.columns)


def test_predefined_cv_rejects_test_rows(syn_analysis):
    s = load_splits(True, syn_analysis)
    train, test = split_frame(syn_analysis, s)
    assert predefined_cv(train, s).get_n_splits() == s.n_folds
    with pytest.raises(ValueError, match="training"):
        predefined_cv(test, s)


def test_baselines():
    X = pd.DataFrame({"treatment": ["A", "A", "B", "B"]})
    r = ArmMeanRegressor().fit(X, [1.0, 3.0, 10.0, 20.0])
    assert r.predict(X).tolist() == [2.0, 2.0, 15.0, 15.0]
    c = ArmRateClassifier().fit(X, [0, 1, 1, 1])
    assert np.allclose(c.predict_proba(X)[:, 1], [0.5, 0.5, 1.0, 1.0])


def test_classification_metrics_toy():
    m = models.classification_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]))
    assert m["TP"] == 1 and m["FN"] == 1 and m["FP"] == 1 and m["TN"] == 1
    assert m["Recall"] == 0.5 and m["Specificity"] == 0.5 and m["Accuracy"] == 0.5


def test_knn_structures_agree():
    rng = np.random.default_rng(4)
    t = knn_timing.compare(
        rng.normal(size=(200, 5)), rng.normal(size=(30, 5)), ["kd_tree", "ball_tree", "brute"], 5, 2
    )
    assert (t["max_abs_distance_diff_vs_brute"] < 1e-10).all()


@pytest.mark.parametrize("stage", ["represent", "supervised"])
def test_stage_runs_on_synthetic(stage, fast_config):
    import importlib

    importlib.import_module(f"equipoise.{stage}.stage").run(True)
    prefix = {"represent": "m3", "supervised": "m5"}[stage]
    assert list((fast_config / "reports" / "figures").glob(f"{prefix}_*.png"))
    assert list((fast_config / "reports" / "tables").glob(f"{prefix}_*.md"))

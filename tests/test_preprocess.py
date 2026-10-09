import numpy as np
import pandas as pd
import scipy.sparse as sp

from equipoise.data.features import assert_no_leakage
from equipoise.data.splits import load_splits, split_frame
from equipoise.preprocess import missing, outliers
from equipoise.preprocess.discretize import discretize
from equipoise.preprocess.encode import build_preprocessor, onehot_sparse, transform_frame
from equipoise.preprocess.harmonize import MachineZScore


def _train_test(df):
    return split_frame(df, load_splits(True, df))


def test_preprocessor_fitted_on_train_only(syn_analysis, cfg):
    train, test = _train_test(syn_analysis)
    ct = build_preprocessor(cfg).fit(train)
    num = ct.named_transformers_["num"]
    scaler, imputer = num.named_steps["scale"], num.named_steps["impute"]
    # scaler statistics are the training means, not the full-data means
    assert np.isclose(
        scaler.mean_[cfg["features"]["continuous"].index("X_age")], train["X_age"].mean()
    )
    assert not np.isclose(scaler.mean_[0], syn_analysis["X_age"].mean())
    # the KNN imputer only stores training rows
    assert imputer._fit_X.shape[0] == len(train)
    harm = ct.named_transformers_["cst"].named_steps["harmonize"]
    assert harm.stats_["__pooled__"][2] == train["X_cst_um"].notna().sum()


def test_preprocessor_output_is_complete_and_leak_free(syn_analysis, cfg):
    train, test = _train_test(syn_analysis)
    ct = build_preprocessor(cfg).fit(train)
    Z = transform_frame(ct, test)
    assert Z.notna().all().all() and len(Z) == len(test)
    assert_no_leakage(Z.columns)
    # an outcome column in the input must simply be ignored
    Z2 = transform_frame(ct, test.assign(Y_va_change_wk52=999.0))
    pd.testing.assert_frame_equal(Z, Z2)


def test_log_transform_applied(syn_analysis, cfg):
    ct = build_preprocessor(cfg).fit(syn_analysis)
    scaler = ct.named_transformers_["num"].named_steps["scale"]
    i = cfg["features"]["continuous"].index("X_hba1c")
    assert np.isclose(scaler.mean_[i], np.log(syn_analysis["X_hba1c"]).mean())


def test_machine_zscore():
    df = pd.DataFrame(
        {"cst": [300, 320, 340, 400, 420, 440, 500.0], "m": ["A", "A", "A", "B", "B", "B", "C"]}
    )
    h = MachineZScore(min_group_n=3).fit(df)
    z = h.transform(df).ravel()
    assert np.allclose(z[:3], [-1, 0, 1]) and np.allclose(z[3:6], [-1, 0, 1])
    pooled_mean, pooled_sd, _ = h.stats_["__pooled__"]
    assert np.isclose(z[6], (500 - pooled_mean) / pooled_sd)  # C too small -> pooled
    assert np.isnan(h.transform(pd.DataFrame({"cst": [np.nan], "m": ["A"]}))).all()


def test_onehot_is_sparse(syn_analysis, cfg):
    M, names = onehot_sparse(syn_analysis, cfg)
    assert sp.issparse(M) and M.shape == (len(syn_analysis), len(names))
    assert np.allclose(M.sum(axis=1), len(cfg["features"]["nominal"]))


def test_bands(syn_analysis, cfg):
    b = discretize(syn_analysis, cfg)
    assert set(b.columns) >= {"X_hba1c_band", "X_va_letters_band", "X_age_band"}
    worse = syn_analysis["X_va_letters"] <= cfg["trial"]["va_below_69_cutoff"]
    assert (b.loc[~worse, "X_va_letters_band"] == ">=69 (20/40 or better)").all()


def test_rubin_rules():
    p = missing.rubin_pool(np.array([1.0, 1.0, 1.0]), np.array([0.25, 0.25, 0.25]))
    assert p.estimate == 1 and np.isclose(p.se, 0.5) and p.between_var == 0
    q = np.array([1.0, 2.0, 3.0])
    u = np.array([1.0, 1.0, 1.0])
    p = missing.rubin_pool(q, u)
    assert np.isclose(p.se**2, 1 + (1 + 1 / 3) * 1.0)
    assert np.isclose(p.df, 2 * (1 + 1 / ((4 / 3) / 1)) ** 2)
    assert p.ci_low < 2 < p.ci_high


def test_missing_strategies(syn_analysis, cfg):
    cfg = {
        **cfg,
        "preprocess": {
            **cfg["preprocess"],
            "missing": {**cfg["preprocess"]["missing"], "n_imputations": 3, "mice_max_iter": 3},
        },
    }
    summary, per_imp = missing.compare_strategies(syn_analysis, cfg)
    assert len(summary) == 6 and len(per_imp) == 6
    cc = summary[summary.strategy == "complete-case"].set_index("comparison")["estimate"]
    m = syn_analysis.groupby("treatment")["Y_va_change_wk52"].mean()
    assert np.isclose(cc["Aflibercept - Bevacizumab"], m["Aflibercept"] - m["Bevacizumab"])
    assert (summary["ci_low"] < summary["estimate"]).all()


def test_iqr_flags():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100.0]})
    f = outliers.iqr_flags(df, ["x"], np.ones(6, bool), 1.5)
    assert f["x"].tolist() == [False] * 5 + [True]


def test_outlier_detection_runs(syn_analysis, cfg):
    train_mask = ~syn_analysis.pt_id.isin(load_splits(True, syn_analysis).test_ids).to_numpy()
    res = outliers.detect(syn_analysis, train_mask, cfg)
    assert len(res["flags"]) == len(syn_analysis)
    assert res["overlap"]["n"].sum() == len(syn_analysis)
    assert res["eps"] > 0

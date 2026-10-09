import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from equipoise.data.splits import load_splits, split_frame
from equipoise.preprocess.discretize import discretize
from equipoise.preprocess.encode import build_preprocessor, transform_frame
from equipoise.represent import cluster, rules
from equipoise.represent.pca import compare_with_sklearn, svd_pca


def _Z(df, cfg):
    train, _ = split_frame(df, load_splits(True, df))
    ct = build_preprocessor(cfg).fit(train)
    return train, transform_frame(ct, train)


def test_svd_pca_matches_sklearn(syn_analysis, cfg):
    _, Z = _Z(syn_analysis, cfg)
    diffs = compare_with_sklearn(Z.to_numpy(), 8)
    assert max(diffs.values()) < 1e-8
    p = svd_pca(Z.to_numpy())
    assert np.isclose(p.explained_variance_ratio.sum(), 1.0)
    assert np.allclose(p.explained_variance, PCA().fit(Z.to_numpy()).explained_variance_)


def test_pca_components_orthonormal():
    X = np.random.default_rng(0).normal(size=(100, 6))
    p = svd_pca(X, 4)
    assert np.allclose(p.components @ p.components.T, np.eye(4))


def test_clustering_and_cluster_effects(syn_analysis, cfg):
    train, Z = _Z(syn_analysis, cfg)
    cfg = {**cfg, "represent": {**cfg["represent"], "k_range": [2, 3]}}
    res = cluster.fit_clusterings(Z.to_numpy(), cfg)
    assert res["k_kmeans"] in (2, 3) and len(res["kmeans"]) == len(train)
    eff = cluster.effects_by_cluster(train, res["kmeans"], "kmeans", cfg)
    assert set(eff["comparison"]) == {"Aflibercept - Bevacizumab", "Ranibizumab - Bevacizumab"}
    assert (eff["note"].str.contains("exploratory")).all()


def test_apriori_rules_target_outcome_only(syn_analysis, cfg):
    train, _ = _Z(syn_analysis, cfg)
    with_harm = train.assign(X_cst_um_harm=0.0)
    r = rules.rules_by_arm(train, discretize(with_harm, cfg), cfg)
    assert not r.empty
    assert (r["consequent"] == rules.OUTCOME_ITEM).all()
    assert not r["antecedents"].str.contains(rules.OUTCOME_ITEM, regex=False).any()
    assert (r["support"] >= cfg["represent"]["apriori"]["min_support"] - 1e-12).all()
    assert (r["confidence"] >= cfg["represent"]["apriori"]["min_confidence"] - 1e-12).all()


def test_baskets_outcome_item():
    df = pd.DataFrame({"Y_va_change_wk52": [20.0, 3.0], "X_female": [1, 0]})
    cfg = {
        "represent": {"apriori": {"binary_items": ["X_female"]}},
        "analysis": {"responder_threshold_letters": 15, "primary_outcome": "Y_va_change_wk52"},
    }
    b = rules.baskets(df, pd.DataFrame({"band": ["a", "b"]}), cfg)
    assert b[rules.OUTCOME_ITEM].tolist() == [True, False]
    assert b["X_female=1"].tolist() == [True, False]

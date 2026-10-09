import numpy as np
import pytest

from equipoise.data.load import analysis_path
from equipoise.data.splits import (
    build_splits,
    cv_folds,
    load_splits,
    save_splits,
    split_frame,
    splits_path,
)
from equipoise.tracking import file_hash


def test_saved_synthetic_split_is_current(syn_analysis, cfg):
    saved = load_splits(True, syn_analysis)
    fresh = build_splits(syn_analysis, file_hash(analysis_path(True)), cfg)
    assert saved == fresh, (
        "data/synthetic/splits.json is stale: run `python -m equipoise split --synthetic`"
    )


def test_folds_partition_training_set_and_are_stratified(syn_analysis, cfg):
    s = build_splits(syn_analysis, "x", cfg)
    train, test = split_frame(syn_analysis, s)
    assert set(train.pt_id) == set(s.train_ids) and set(test.pt_id) == set(s.test_ids)
    seen = []
    arm = syn_analysis.set_index("pt_id")["treatment"]
    for k in range(s.n_folds):
        fit, val = s.fold_ids(k)
        assert set(fit).isdisjoint(val) and not set(val) & set(s.test_ids)
        seen += val
        share = arm.loc[val].value_counts(normalize=True)
        overall = arm.loc[s.train_ids].value_counts(normalize=True)
        assert np.allclose(share.sort_index(), overall.sort_index(), atol=0.02)
    assert sorted(seen) == s.train_ids


def test_cv_folds_reproducible(syn_analysis, cfg):
    a = [v.tolist() for _, v in cv_folds(syn_analysis, cfg=cfg)]
    b = [v.tolist() for _, v in cv_folds(syn_analysis, cfg=cfg)]
    assert a == b


def test_roundtrip_and_mismatch(tmp_path, syn_analysis, cfg):
    s = build_splits(syn_analysis, "x", cfg)
    p = save_splits(s, tmp_path / "splits.json")
    assert p.exists()
    assert splits_path(True).name == "splits.json"
    with pytest.raises(ValueError):
        import equipoise.data.splits as m

        m_path = m.splits_path
        try:
            m.splits_path = lambda synthetic, cfg=None: p
            load_splits(True, syn_analysis.iloc[:10])
        finally:
            m.splits_path = m_path

import numpy as np

from equipoise.data.splits import make_test_split


def test_same_seed_same_split(syn_analysis):
    assert make_test_split(syn_analysis, seed=42) == make_test_split(syn_analysis, seed=42)


def test_different_seed_different_split(syn_analysis):
    assert make_test_split(syn_analysis, seed=42)[1] != make_test_split(syn_analysis, seed=7)[1]


def test_split_partition_and_size(syn_analysis, cfg):
    train, test = make_test_split(syn_analysis)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(syn_analysis["pt_id"])
    assert abs(len(test) - cfg["split"]["test_fraction"] * len(syn_analysis)) <= 1


def test_split_stratified_by_arm(syn_analysis, cfg):
    _, test = make_test_split(syn_analysis)
    frac = cfg["split"]["test_fraction"]
    arm_n = syn_analysis["treatment"].value_counts()
    test_n = syn_analysis[syn_analysis["pt_id"].isin(test)]["treatment"].value_counts()
    for arm, n in arm_n.items():
        assert abs(test_n[arm] - frac * n) <= 1, arm
    assert np.isclose(test_n.sum(), len(test))

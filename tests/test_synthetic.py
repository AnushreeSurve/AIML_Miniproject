import numpy as np
import pandas as pd

from equipoise.config import load_config
from equipoise.data.synthetic import generate


def _regen():
    cfg = load_config()
    return generate(
        load_config("synthetic"),
        va_cutoff=cfg["trial"]["va_below_69_cutoff"],
        reference_arm=cfg["analysis"]["reference_arm"],
    )


def test_committed_files_are_reproducible(syn_analysis, syn_visits):
    data = _regen()
    pd.testing.assert_frame_equal(data.analysis, syn_analysis, check_dtype=False, atol=1e-9)
    pd.testing.assert_frame_equal(data.visits, syn_visits, check_dtype=False, atol=1e-9)


def test_planted_effect_is_visible(syn_analysis):
    m = syn_analysis.groupby(["X_va_below_69", "treatment"])["Y_va_change_wk52"].mean()
    low = m[1, "Aflibercept"] - m[1, "Bevacizumab"]
    high = m[0, "Aflibercept"] - m[0, "Bevacizumab"]
    assert low - high > 3, (low, high)


def test_truth_effects(syn_truth, cfg):
    p = load_config("synthetic")["outcome_wk52"]
    below = syn_truth["pt_id"].map(_regen().analysis.set_index("pt_id")["X_va_below_69"]).to_numpy()
    expected = p["afl_main"] + p["afl_extra_if_va_below_69"] * below
    assert np.allclose(syn_truth["tau_Aflibercept"], expected)


def test_visits_agree_with_outcomes(syn_analysis, syn_visits):
    v = syn_visits
    base = v[v.visit_label == "baseline"].set_index("pt_id")["va_letters"]
    wk52 = v[v.visit_label == "wk52"].set_index("pt_id")["va_letters"]
    a = syn_analysis.set_index("pt_id")
    has = a.index[a["M_has_wk52"] == 1]
    assert set(wk52.index) == set(has)
    assert np.allclose(wk52[has] - base[has], a.loc[has, "Y_va_change_wk52"])
    yr1 = v[v.visit_label.isin(["baseline"] + [f"wk{w}" for w in range(4, 52, 4)])]
    inj = yr1.groupby("pt_id")["injected"].sum().reindex(a.index, fill_value=0)
    assert (inj == a["Y_n_inj_yr1"]).all()
    assert (v.groupby("pt_id")["injected"].sum() == a["Y_n_inj_total"]).all()

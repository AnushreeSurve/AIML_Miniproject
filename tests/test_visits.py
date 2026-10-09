import numpy as np
import pandas as pd
import pytest

from equipoise.data.schema import validate_visits
from equipoise.data.visits import (
    VisitAnomaly,
    assemble_visit_table,
    attach_flag,
    attach_measurements,
    cross_check_change,
    cross_check_count,
    nearest_visit,
    year1_mask,
)

V = pd.DataFrame({"pt_id": [1, 1, 1, 2, 2], "days_from_rand": [0, 28, 56, 0, 30]})


def test_nearest_visit_within_window():
    ev = pd.DataFrame({"pt_id": [1, 1, 2, 2], "days_from_rand": [3, 40, 29, 90], "x": [1, 2, 3, 4]})
    kept, rep = nearest_visit(V, ev, window_days=5, name="t")
    assert kept[["pt_id", "visit_day", "x"]].values.tolist() == [[1, 0, 1], [2, 30, 3]]
    assert rep.n_unmatched == 2 and rep.max_gap_days == 3


def test_nearest_visit_never_matches_other_patient():
    ev = pd.DataFrame({"pt_id": [3], "days_from_rand": [0]})
    kept, rep = nearest_visit(V, ev, window_days=100, name="t")
    assert kept.empty and rep.n_unmatched == 1


def test_collapsed_measurements_keep_nearest():
    m = pd.DataFrame(
        {
            "pt_id": [1, 1],
            "days_from_rand": [26, 29],
            "cst_um": [300.0, 310.0],
            "oct_machine": ["Zeiss Cirrus"] * 2,
        }
    )
    out, rep = attach_measurements(V, m, ["cst_um", "oct_machine"], 7, "oct")
    assert out.loc[out.days_from_rand == 28, "cst_um"].item() == 310.0
    assert rep.n_collapsed == 1 and out["cst_um"].notna().sum() == 1


def test_unmatched_injection_fails_loudly():
    ev = pd.DataFrame({"pt_id": [1], "days_from_rand": [14]})
    with pytest.raises(VisitAnomaly, match="no visit"):
        attach_flag(V, ev, "injected", window_days=3)
    out, _ = attach_flag(V, ev, "injected", window_days=3, allow_unmatched=True)
    assert out["injected"].sum() == 0


def test_duplicate_visits_rejected():
    with pytest.raises(VisitAnomaly):
        nearest_visit(pd.concat([V, V.head(1)]), V, 1, "t")


def _decompose(visits, rng, shift_days):
    """Split the synthetic visit table into raw-like normalized frames with date offsets."""
    vi = visits[["pt_id", "days_from_rand", "visit_label"]]
    va = visits[["pt_id", "days_from_rand", "va_letters"]]
    oct_ = visits.dropna(subset=["cst_um"])[["pt_id", "days_from_rand", "cst_um", "oct_machine"]]
    oct_ = oct_.assign(
        days_from_rand=oct_.days_from_rand + rng.integers(0, shift_days + 1, len(oct_))
    )
    inj = visits[visits.injected == 1][["pt_id", "days_from_rand"]]
    inj = inj.assign(days_from_rand=inj.days_from_rand + rng.integers(-1, 2, len(inj)).clip(0))
    las = visits[visits.laser == 1][["pt_id", "days_from_rand"]]
    return vi, va, oct_, inj, las


def test_assemble_roundtrip_reproduces_synthetic_visits(syn_visits, syn_analysis, cfg):
    rng = np.random.default_rng(0)
    vi, va, oct_, inj, las = _decompose(syn_visits, rng, shift_days=5)
    built = assemble_visit_table(
        visit_info=vi,
        va=va,
        oct_=oct_,
        injections=inj,
        laser=las,
        analysis=syn_analysis,
        oct_window_days=cfg["visits"]["oct_match_window_days"],
        event_window_days=cfg["visits"]["event_match_window_days"],
    )
    validate_visits(built.visits, analysis=syn_analysis)
    pd.testing.assert_frame_equal(built.visits, syn_visits, check_dtype=False)
    assert all(r.n_unmatched == 0 for r in built.reports)


def test_cross_check_agrees_on_synthetic(syn_visits, syn_analysis, cfg):
    v = cfg["visits"]
    for wk, label in (("52", v["wk52_label"]), ("104", v["wk104_label"])):
        bad = cross_check_change(
            syn_visits,
            syn_analysis,
            outcome=f"Y_va_change_wk{wk}",
            label=label,
            baseline_label=v["baseline_label"],
            tolerance=v["crosscheck_tolerance_letters"],
        )
        assert bad.empty, bad
    yr1 = cross_check_count(
        syn_visits,
        syn_analysis,
        flag="injected",
        outcome="Y_n_inj_yr1",
        mask=year1_mask(syn_visits, v["wk52_label"], v["year1_end_day"]),
    )
    assert yr1.empty
    assert cross_check_count(
        syn_visits, syn_analysis, flag="injected", outcome="Y_n_inj_total"
    ).empty


def test_cross_check_reports_mismatch(syn_visits, syn_analysis, cfg):
    v = cfg["visits"]
    bad_analysis = syn_analysis.copy()
    i = bad_analysis.index[bad_analysis.M_has_wk52 == 1][:3]
    bad_analysis.loc[i, "Y_va_change_wk52"] += 5
    bad = cross_check_change(
        syn_visits,
        bad_analysis,
        outcome="Y_va_change_wk52",
        label=v["wk52_label"],
        baseline_label=v["baseline_label"],
        tolerance=v["crosscheck_tolerance_letters"],
    )
    assert sorted(bad.pt_id) == sorted(bad_analysis.loc[i, "pt_id"])

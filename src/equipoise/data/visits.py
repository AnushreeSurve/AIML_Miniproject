"""Visit-table assembly: align measurements and events to study visits.

The long visit table has one row per patient x visit for the study eye. OCT
gradings and treatment events are recorded separately from vision tests, so
each OCT reading is attached to the *nearest* visit of the same patient
within a configurable window (``visits.oct_match_window_days``). Injections
and laser sessions become 0/1 flags on the visit they are nearest to
(``visits.event_match_window_days``). Anything that cannot be matched is
counted and reported, never silently dropped. ``cross_check_change`` then
confirms that the week-52 vision change derived from the visit table equals
``Y_va_change_wk52`` in the analysis table.

These functions work on already-normalized frames (``pt_id``,
``days_from_rand``, values), so they are tested on synthetic data and do
not depend on raw DRCR column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEY = ["pt_id", "days_from_rand"]


class VisitAnomaly(ValueError):
    """Raised when raw visit data has a problem that must not be ignored."""


@dataclass
class MatchReport:
    """Counts from one nearest-visit matching step."""

    name: str
    n_events: int
    n_matched: int
    n_unmatched: int
    n_collapsed: int  # events that shared a visit with a nearer event
    max_gap_days: int
    unmatched: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def as_row(self) -> dict[str, int | str]:
        """Summary row for the build report table."""
        return {
            "step": self.name,
            "events": self.n_events,
            "matched": self.n_matched,
            "unmatched": self.n_unmatched,
            "collapsed_onto_same_visit": self.n_collapsed,
            "max_gap_days": self.max_gap_days,
        }


def _check_keys(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise VisitAnomaly(f"{name}: missing key columns {missing}")
    if df[KEY].isna().any().any():
        raise VisitAnomaly(
            f"{name}: {int(df[KEY].isna().any(axis=1).sum())} rows with missing keys"
        )


def nearest_visit(
    visits: pd.DataFrame, events: pd.DataFrame, window_days: int, name: str
) -> tuple[pd.DataFrame, MatchReport]:
    """Map each event to its patient's nearest visit day within ``window_days``.

    Returns the matched events with a ``visit_day`` and ``gap_days`` column.
    When several events land on one visit, only the nearest is kept (ties:
    the earlier event), and the number collapsed is reported.
    """
    _check_keys(visits, "visits")
    _check_keys(events, name)
    if visits.duplicated(KEY).any():
        raise VisitAnomaly("visits: duplicated (pt_id, days_from_rand)")
    v = (
        visits[KEY]
        .rename(columns={"days_from_rand": "visit_day"})
        .astype({"visit_day": "int64"})
        .sort_values("visit_day")
    )
    e = events.copy()
    e["days_from_rand"] = e["days_from_rand"].astype("int64")
    e = e.sort_values("days_from_rand").reset_index(drop=True)
    m = pd.merge_asof(
        e,
        v,
        left_on="days_from_rand",
        right_on="visit_day",
        by="pt_id",
        direction="nearest",
        tolerance=window_days,
    )
    matched = m[m["visit_day"].notna()].copy()
    unmatched = m[m["visit_day"].isna()].drop(columns=["visit_day"])
    matched["visit_day"] = matched["visit_day"].astype("int64")
    matched["gap_days"] = (matched["days_from_rand"] - matched["visit_day"]).abs()
    matched = matched.sort_values(["pt_id", "visit_day", "gap_days", "days_from_rand"])
    kept = matched.drop_duplicates(["pt_id", "visit_day"], keep="first")
    report = MatchReport(
        name=name,
        n_events=len(e),
        n_matched=len(matched),
        n_unmatched=len(unmatched),
        n_collapsed=len(matched) - len(kept),
        max_gap_days=int(kept["gap_days"].max()) if len(kept) else 0,
        unmatched=unmatched,
    )
    return kept.reset_index(drop=True), report


def attach_measurements(
    visits: pd.DataFrame,
    measurements: pd.DataFrame,
    columns: list[str],
    window_days: int,
    name: str,
) -> tuple[pd.DataFrame, MatchReport]:
    """Add ``columns`` from ``measurements`` to the nearest visit (NaN if none in window)."""
    kept, report = nearest_visit(visits, measurements, window_days, name)
    add = kept[["pt_id", "visit_day", *columns]].rename(columns={"visit_day": "days_from_rand"})
    out = visits.drop(columns=[c for c in columns if c in visits.columns]).merge(
        add, on=KEY, how="left", validate="one_to_one"
    )
    return out, report


def attach_flag(
    visits: pd.DataFrame,
    events: pd.DataFrame,
    flag: str,
    window_days: int,
    *,
    allow_unmatched: bool = False,
) -> tuple[pd.DataFrame, MatchReport]:
    """Add a 0/1 ``flag`` column: 1 if an event is nearest to that visit.

    Unmatched events (e.g. an injection with no visit nearby) raise
    ``VisitAnomaly`` unless ``allow_unmatched`` is set, because dropping them
    would silently change injection counts.
    """
    kept, report = nearest_visit(visits, events[KEY], window_days, flag)
    if report.n_unmatched and not allow_unmatched:
        raise VisitAnomaly(
            f"{flag}: {report.n_unmatched} events have no visit within {window_days} days:\n"
            f"{report.unmatched.head(10).to_string(index=False)}"
        )
    hit = kept[["pt_id", "visit_day"]].rename(columns={"visit_day": "days_from_rand"})
    hit[flag] = 1
    out = visits.drop(columns=[flag], errors="ignore").merge(hit, on=KEY, how="left")
    out[flag] = out[flag].fillna(0).astype(int)
    return out, report


def derive_change(
    visits: pd.DataFrame, label: str, baseline_label: str, value: str = "va_letters"
) -> pd.Series:
    """Per-patient ``value`` at visit ``label`` minus its baseline value."""
    by = visits.dropna(subset=["visit_label"])
    for lab in (label, baseline_label):
        dup = by[by["visit_label"] == lab]["pt_id"].duplicated()
        if dup.any():
            raise VisitAnomaly(f"{int(dup.sum())} patients have more than one '{lab}' visit")
    at = by[by["visit_label"] == label].set_index("pt_id")[value]
    base = by[by["visit_label"] == baseline_label].set_index("pt_id")[value]
    return (at - base.reindex(at.index)).rename(f"{value}_change_{label}")


def cross_check_change(
    visits: pd.DataFrame,
    analysis: pd.DataFrame,
    *,
    outcome: str,
    label: str,
    baseline_label: str,
    tolerance: float,
    value: str = "va_letters",
) -> pd.DataFrame:
    """Compare the visit-derived change with ``analysis[outcome]``.

    Returns one row per mismatching patient (different value beyond
    ``tolerance``, or present in one table and missing in the other).
    An empty frame means the tables agree.
    """
    derived = derive_change(visits, label, baseline_label, value)
    expected = analysis.set_index("pt_id")[outcome]
    both = pd.DataFrame({"derived": derived.reindex(expected.index), "expected": expected})
    both["diff"] = both["derived"] - both["expected"]
    one_missing = both["derived"].isna() != both["expected"].isna()
    differ = both["diff"].abs() > tolerance
    bad = both[one_missing | differ.fillna(False)].reset_index()
    return bad


def year1_mask(visits: pd.DataFrame, wk52_label: str, year1_end_day: int) -> pd.Series:
    """Flag year-1 visits: those before the patient's week-52 visit.

    Patients without a week-52 visit fall back to ``days_from_rand <
    year1_end_day``. The week-52 visit itself starts year 2, which is why at
    most 13 injections (weeks 0-48) can count as year 1.
    """
    wk52_day = visits[visits["visit_label"] == wk52_label].set_index("pt_id")["days_from_rand"]
    cutoff = visits["pt_id"].map(wk52_day).fillna(year1_end_day)
    return visits["days_from_rand"] < cutoff


def cross_check_count(
    visits: pd.DataFrame,
    analysis: pd.DataFrame,
    *,
    flag: str,
    outcome: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Compare per-patient sums of ``flag`` (rows where ``mask``) with ``analysis[outcome]``."""
    v = visits if mask is None else visits[mask]
    counts = v.groupby("pt_id")[flag].sum()
    expected = analysis.set_index("pt_id")[outcome]
    both = pd.DataFrame(
        {"derived": counts.reindex(expected.index, fill_value=0), "expected": expected}
    )
    return both[both["derived"] != both["expected"]].reset_index()


def summarize_visits(visits: pd.DataFrame) -> pd.DataFrame:
    """Small descriptive table of the visit table, for the build report."""
    per_pt = visits.groupby("pt_id").size()
    return pd.DataFrame(
        {
            "metric": [
                "patients",
                "visit rows",
                "visits per patient (median)",
                "rows with vision",
                "rows with CST",
                "injection visits",
                "laser visits",
                "max days from randomization",
            ],
            "value": [
                visits["pt_id"].nunique(),
                len(visits),
                float(np.median(per_pt)),
                int(visits["va_letters"].notna().sum()),
                int(visits["cst_um"].notna().sum()),
                int(visits["injected"].sum()),
                int(visits["laser"].sum()),
                int(visits["days_from_rand"].max()),
            ],
        }
    )


@dataclass
class AssembledVisits:
    """Result of ``assemble_visit_table``: the table plus its matching reports."""

    visits: pd.DataFrame
    reports: list[MatchReport]


def assemble_visit_table(
    *,
    visit_info: pd.DataFrame,
    va: pd.DataFrame,
    oct_: pd.DataFrame,
    injections: pd.DataFrame,
    laser: pd.DataFrame,
    analysis: pd.DataFrame,
    oct_window_days: int,
    event_window_days: int,
) -> AssembledVisits:
    """Build the study-eye visit table from normalized, study-eye-only frames.

    ``visit_info`` gives the visit skeleton (``pt_id, days_from_rand,
    visit_label``); ``va`` must have exactly one vision value per visit day
    (``va_letters``); ``oct_`` has ``cst_um, oct_machine``; ``injections``
    and ``laser`` have one row per event. Only randomized patients in
    ``analysis`` are kept and ``treatment`` comes from ``analysis``.
    """
    from equipoise.data.schema import VISIT_COLUMNS

    pts = set(analysis["pt_id"])
    skel = visit_info[visit_info["pt_id"].isin(pts)][[*KEY, "visit_label"]]
    if skel.duplicated(KEY).any():
        raise VisitAnomaly(f"{int(skel.duplicated(KEY).sum())} duplicated visits in visit_info")
    va = va[va["pt_id"].isin(pts)]
    if va.duplicated(KEY).any():
        raise VisitAnomaly(f"{int(va.duplicated(KEY).sum())} visit days with more than one VA test")
    out = skel.merge(va[[*KEY, "va_letters"]], on=KEY, how="left")
    orphan_va = len(va) - int(out["va_letters"].notna().sum())
    reports = [
        MatchReport("va (same day)", len(va), len(va) - orphan_va, orphan_va, 0, 0),
    ]
    out, r = attach_measurements(
        out, oct_[oct_["pt_id"].isin(pts)], ["cst_um", "oct_machine"], oct_window_days, "oct"
    )
    reports.append(r)
    for name, ev in (("injected", injections), ("laser", laser)):
        out, r = attach_flag(out, ev[ev["pt_id"].isin(pts)], name, event_window_days)
        reports.append(r)
    out["treatment"] = out["pt_id"].map(analysis.set_index("pt_id")["treatment"])
    out = out[[c.name for c in VISIT_COLUMNS]].sort_values(KEY).reset_index(drop=True)
    return AssembledVisits(out, reports)

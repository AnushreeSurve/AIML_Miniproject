"""Build ``data/protocolT_visits.csv``: one row per patient x visit, study eye only.

Raw DRCR column names come from ``config/raw_tables.yaml``, which must be
filled by inspecting the real files; the script refuses to run while any
entry is null. Study-eye selection reuses the M0 decision by taking each
patient's ``study_eye`` from ``data/protocolT_analysis.csv``. OCT gradings
are matched to the nearest visit within ``visits.oct_match_window_days``;
injections and laser within ``visits.event_match_window_days``. The script
fails loudly on anomalies and cross-checks the week-52 / week-104 vision
change and injection counts against the analysis table.

Usage::

    python scripts/build_protocolT_visits.py "data/raw/Data Tables - Text files" \
        data/protocolT_visits.csv [--max-mismatches 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from equipoise.config import load_config, resolve_path
from equipoise.data.load import load_analysis
from equipoise.data.schema import validate_visits
from equipoise.data.visits import (
    assemble_visit_table,
    cross_check_change,
    cross_check_count,
    summarize_visits,
    year1_mask,
)
from equipoise.tracking import save_table

MODULE = "m0"


def _null_entries(d: Any, prefix: str = "") -> list[str]:
    if isinstance(d, dict):
        return [p for k, v in d.items() for p in _null_entries(v, f"{prefix}{k}.")]
    return [prefix.rstrip(".")] if d is None else []


def read_raw(raw_dir: Path, mapping: dict[str, Any], table: str) -> pd.DataFrame:
    """Read one raw table and rename its mapped columns to the normalized names."""
    spec = mapping["tables"][table]
    path = raw_dir / f"{spec['file']}{mapping['file_extension']}"
    if not path.exists():
        sys.exit(f"raw table not found: {path}")
    raw = pd.read_csv(
        path, sep=mapping["delimiter"], encoding=mapping["encoding"], low_memory=False
    )
    cols = spec["columns"]
    missing = [c for c in cols.values() if c not in raw.columns]
    if missing:
        sys.exit(f"{path.name}: mapped columns not found: {missing}")
    return raw[list(cols.values())].rename(columns={v: k for k, v in cols.items()})


def study_eye_only(df: pd.DataFrame, analysis: pd.DataFrame, codes: dict[str, Any]) -> pd.DataFrame:
    """Keep rows for each patient's study eye (as decided by M0)."""
    inv = {str(v): k for k, v in codes.items()}
    eye = df["eye"].astype(str).map(inv)
    if eye.isna().any():
        sys.exit(f"unknown eye codes: {sorted(df.loc[eye.isna(), 'eye'].astype(str).unique())}")
    study = df["pt_id"].map(analysis.set_index("pt_id")["study_eye"])
    return df[eye.to_numpy() == study.to_numpy()].drop(columns=["eye"])


def main(argv: list[str] | None = None) -> None:
    """Build, validate and cross-check the visit table."""
    cfg = load_config()
    vcfg = cfg["visits"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_dir", type=Path, nargs="?", default=resolve_path(cfg["paths"]["raw_dir"]))
    ap.add_argument(
        "out_csv", type=Path, nargs="?", default=resolve_path(cfg["paths"]["visits_csv"])
    )
    ap.add_argument(
        "--max-mismatches",
        type=int,
        default=0,
        help="allowed week-52 vision-change mismatches before failing",
    )
    args = ap.parse_args(argv)

    mapping = load_config("raw_tables")
    nulls = _null_entries(mapping)
    if nulls:
        sys.exit(
            "config/raw_tables.yaml is not filled in (inspect the raw files first):\n  "
            + "\n  ".join(nulls)
        )
    if not args.raw_dir.is_dir():
        sys.exit(f"raw directory not found: {args.raw_dir}")

    analysis = load_analysis(synthetic=False, cfg=cfg)
    labels = {str(v): k for k, v in mapping["visit_labels"].items()}
    vi = read_raw(args.raw_dir, mapping, "visit_info")
    vi["visit_label"] = vi["visit_label"].astype(str).map(lambda x: labels.get(x, x))
    frames = {
        t: study_eye_only(read_raw(args.raw_dir, mapping, t), analysis, mapping["eye_codes"])
        for t in ("va", "oct", "injections", "laser")
    }
    built = assemble_visit_table(
        visit_info=vi,
        va=frames["va"],
        oct_=frames["oct"],
        injections=frames["injections"],
        laser=frames["laser"],
        analysis=analysis,
        oct_window_days=vcfg["oct_match_window_days"],
        event_window_days=vcfg["event_match_window_days"],
    )
    visits = validate_visits(built.visits, analysis=analysis)
    report = pd.DataFrame([r.as_row() for r in built.reports])
    save_table(report, MODULE, "visits_matching", log=False)
    save_table(summarize_visits(visits), MODULE, "visits_summary", log=False)

    checks = {
        "va_change_wk52": cross_check_change(
            visits,
            analysis,
            outcome="Y_va_change_wk52",
            label=vcfg["wk52_label"],
            baseline_label=vcfg["baseline_label"],
            tolerance=vcfg["crosscheck_tolerance_letters"],
        ),
        "va_change_wk104": cross_check_change(
            visits,
            analysis,
            outcome="Y_va_change_wk104",
            label=vcfg["wk104_label"],
            baseline_label=vcfg["baseline_label"],
            tolerance=vcfg["crosscheck_tolerance_letters"],
        ),
        "n_inj_yr1": cross_check_count(
            visits,
            analysis,
            flag="injected",
            outcome="Y_n_inj_yr1",
            mask=year1_mask(visits, vcfg["wk52_label"], vcfg["year1_end_day"]),
        ),
        "n_inj_total": cross_check_count(
            visits, analysis, flag="injected", outcome="Y_n_inj_total"
        ),
    }
    summary = pd.DataFrame([{"check": k, "mismatches": len(v)} for k, v in checks.items()])
    save_table(summary, MODULE, "visits_crosscheck", log=False)
    for k, v in checks.items():
        if len(v):
            save_table(v, MODULE, f"visits_mismatch_{k}", log=False)
    print(report.to_string(index=False))
    print(summary.to_string(index=False))
    if len(checks["va_change_wk52"]) > args.max_mismatches:
        sys.exit(
            f"{len(checks['va_change_wk52'])} week-52 mismatches (> {args.max_mismatches}); "
            "see reports/tables/m0_visits_mismatch_va_change_wk52.csv"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    visits.to_csv(args.out_csv, index=False)
    print(
        f"wrote {len(visits)} visit rows for {visits['pt_id'].nunique()} patients -> {args.out_csv}"
    )


if __name__ == "__main__":
    main()

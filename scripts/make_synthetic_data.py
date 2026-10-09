"""Write the synthetic Protocol T-like tables to ``data/synthetic/``.

The schema (column names, order, types, category levels) comes from
``equipoise.data.schema``, which was frozen from the real M0 table. If the
real table is present, this script first checks that the real table still
matches that schema and stops if it has drifted, so the synthetic data can
never silently diverge from the real one. Output depends only on
``config/synthetic.yaml`` (including its seed), never on the real data.

Usage::

    python scripts/make_synthetic_data.py [--out-dir data/synthetic] [--n 660] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from equipoise.config import get_path, load_config, resolve_path
from equipoise.data.load import ANALYSIS_FILE, TRUTH_FILE, VISITS_FILE
from equipoise.data.schema import (
    ANALYSIS_COLUMNS,
    ANALYSIS_SPEC,
    VISIT_SPEC,
    coerce_dtypes,
    validate_analysis,
    validate_visits,
)
from equipoise.data.synthetic import generate


def check_real_schema(path: Path) -> None:
    """Fail if the real analysis table no longer matches the frozen schema."""
    real = pd.read_csv(path, nrows=None)
    expected = [c.name for c in ANALYSIS_COLUMNS]
    if list(real.columns) != expected:
        sys.exit(
            f"Real table {path} columns differ from schema.py; update the schema first.\n"
            f"real-only: {sorted(set(real.columns) - set(expected))}\n"
            f"schema-only: {sorted(set(expected) - set(real.columns))}"
        )
    validate_analysis(coerce_dtypes(real, ANALYSIS_SPEC), expect_trial=True)
    print(f"real table {path.name}: schema matches ({len(real)} rows)")


def main(argv: list[str] | None = None) -> None:
    """Generate, validate and write the synthetic tables."""
    cfg = load_config()
    params = load_config("synthetic")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=resolve_path(cfg["paths"]["synthetic_dir"]))
    ap.add_argument("--n", type=int, default=params["n_patients"])
    ap.add_argument("--seed", type=int, default=params["seed"])
    args = ap.parse_args(argv)
    params.update(n_patients=args.n, seed=args.seed)

    real_path = get_path("analysis_csv", cfg)
    if real_path.exists():
        check_real_schema(real_path)

    data = generate(
        params,
        va_cutoff=cfg["trial"]["va_below_69_cutoff"],
        reference_arm=cfg["analysis"]["reference_arm"],
    )
    validate_analysis(data.analysis, trial=cfg["trial"])
    validate_visits(data.visits, analysis=data.analysis)
    assert set(VISIT_SPEC) == set(data.visits.columns)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.analysis.to_csv(args.out_dir / ANALYSIS_FILE, index=False)
    data.visits.to_csv(args.out_dir / VISITS_FILE, index=False)
    data.truth.to_csv(args.out_dir / TRUTH_FILE, index=False)
    counts = data.analysis["treatment"].value_counts().to_dict()
    print(
        f"wrote {len(data.analysis)} patients {counts}, {len(data.visits)} visits -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()

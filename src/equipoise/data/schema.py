"""Column specifications and validators for the two Equipoise tables.

The analysis table (one row per patient, M0 output) and the visit table (one
row per patient x visit, M7 input) are described column by column: type,
role, whether missing values are allowed, valid range and category levels.
``validate_analysis`` / ``validate_visits`` check a DataFrame against that
specification and raise ``SchemaError`` listing *every* problem found, so
anomalies fail loudly instead of propagating into models.

The specification was frozen from the real M0 table (column names, types and
category labels only - no patient values). Ranges are clinical plausibility
bounds, not observed minima/maxima.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Kind = Literal["int", "float", "str"]
Role = Literal["id", "treatment", "meta", "feature", "outcome", "bookkeeping"]

ARMS: tuple[str, ...] = ("Aflibercept", "Bevacizumab", "Ranibizumab")
BINARY = (0, 1)


class SchemaError(ValueError):
    """Raised when a table does not match its specification."""


@dataclass(frozen=True)
class ColumnSpec:
    """Specification of one column."""

    name: str
    kind: Kind
    role: Role
    description: str
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    categories: tuple[Any, ...] | None = None


def _bin(name: str, desc: str, role: Role = "feature", nullable: bool = False) -> ColumnSpec:
    kind: Kind = "float" if nullable else "int"
    return ColumnSpec(name, kind, role, desc, nullable, categories=BINARY)


# ------------------------------------------------------------ analysis table
ANALYSIS_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("pt_id", "int", "id", "DRCR patient identifier (PtID)", min=1),
    ColumnSpec("treatment", "str", "treatment", "Randomized arm", categories=ARMS),
    ColumnSpec(
        "study_eye", "str", "meta", "Study eye (OD right / OS left)", categories=("OD", "OS")
    ),
    ColumnSpec("X_age", "int", "feature", "Age at randomization (years)", min=18, max=100),
    ColumnSpec("X_bmi", "float", "feature", "Body-mass index (kg/m^2)", True, 10, 80),
    _bin("X_cad", "Coronary artery disease history"),
    ColumnSpec(
        "X_cst_um",
        "float",
        "feature",
        "Central subfield thickness, machine units (um)",
        True,
        100,
        1200,
    ),
    ColumnSpec("X_dbp", "float", "feature", "Diastolic blood pressure (mmHg)", min=30, max=150),
    ColumnSpec(
        "X_diabetes_duration_yrs", "int", "feature", "Diabetes duration (years)", min=0, max=90
    ),
    ColumnSpec(
        "X_diabetes_type",
        "str",
        "feature",
        "Diabetes type",
        categories=("Type 1", "Type 2", "Uncertain"),
    ),
    ColumnSpec(
        "X_dr_severity_clinical",
        "str",
        "feature",
        "Clinical DR severity (ordinal)",
        True,
        categories=(
            "Microaneurysms only",
            "Mild/moderate NPDR",
            "Severe NPDR",
            "PDR and/or prior scatter",
        ),
    ),
    ColumnSpec(
        "X_dr_severity_rc",
        "float",
        "feature",
        "Reading-centre DR severity level (ordinal 1-10)",
        True,
        1,
        10,
    ),
    _bin("X_epiretinal_membrane", "Epiretinal membrane on OCT", nullable=True),
    ColumnSpec(
        "X_fellow_va_letters",
        "float",
        "feature",
        "Fellow-eye vision (ETDRS letters)",
        min=0,
        max=100,
    ),
    _bin("X_female", "Female sex"),
    ColumnSpec("X_hba1c", "float", "feature", "HbA1c (%)", True, 3, 20),
    _bin("X_hispanic", "Hispanic ethnicity"),
    _bin("X_hypertension", "Hypertension history"),
    _bin("X_insulin", "Uses insulin"),
    ColumnSpec("X_iop", "float", "feature", "Intraocular pressure (mmHg)", min=0, max=50),
    ColumnSpec(
        "X_log_uacr", "float", "feature", "log urine albumin/creatinine ratio", True, -5, 15
    ),
    ColumnSpec(
        "X_oct_machine",
        "str",
        "feature",
        "OCT machine used at baseline",
        True,
        categories=("Heidelberg Spectralis", "Zeiss Cirrus", "Zeiss Stratus"),
    ),
    _bin("X_prior_antivegf", "Prior anti-VEGF for DME"),
    _bin("X_prior_dme_trt", "Any prior DME treatment"),
    _bin("X_prior_focal_laser", "Prior focal/grid laser"),
    _bin("X_prior_mi", "Prior myocardial infarction"),
    _bin("X_prior_prp", "Prior panretinal photocoagulation"),
    _bin("X_prior_stroke", "Prior stroke"),
    _bin("X_pseudophakic", "Pseudophakic (prior cataract surgery)"),
    ColumnSpec(
        "X_race",
        "str",
        "feature",
        "Race",
        True,
        categories=(
            "White",
            "Black/African American",
            "Asian",
            "More than one race",
            "Native Hawaiian/Other Pacific Islander",
            "American Indian/Alaskan Native",
        ),
    ),
    _bin("X_renal_disease", "Renal disease history"),
    ColumnSpec("X_sbp", "float", "feature", "Systolic blood pressure (mmHg)", min=60, max=260),
    _bin("X_smoker_current", "Current smoker"),
    _bin("X_subretinal_fluid", "Subretinal fluid on OCT", nullable=True),
    _bin("X_va_below_69", "Baseline vision 20/50 or worse (<= 68 letters)"),
    ColumnSpec(
        "X_va_letters",
        "float",
        "feature",
        "Baseline study-eye vision (ETDRS letters)",
        min=24,
        max=78,
    ),
    ColumnSpec(
        "Y_cst_change_wk104", "float", "outcome", "CST change at 2 years (um)", True, -1200, 1200
    ),
    ColumnSpec(
        "Y_cst_change_wk52", "float", "outcome", "CST change at 1 year (um)", True, -1200, 1200
    ),
    _bin("Y_laser_any", "Rescue laser at any time", role="outcome"),
    _bin("Y_laser_yr1", "Rescue laser by 1 year", role="outcome"),
    ColumnSpec(
        "Y_n_inj_total", "int", "outcome", "Study-eye injections over 2 years", min=0, max=30
    ),
    ColumnSpec("Y_n_inj_yr1", "int", "outcome", "Study-eye injections in year 1", min=0, max=14),
    ColumnSpec(
        "Y_va_change_wk104",
        "float",
        "outcome",
        "Vision change at 2 years (letters)",
        True,
        -100,
        100,
    ),
    ColumnSpec(
        "Y_va_change_wk52",
        "float",
        "outcome",
        "Vision change at 1 year (letters) - PRIMARY",
        True,
        -100,
        100,
    ),
    ColumnSpec(
        "M_final_status",
        "str",
        "bookkeeping",
        "Final study status",
        categories=("Completed", "DropPostOutcome", "DropPreOutcome"),
    ),
    _bin("M_has_wk104", "Week-104 vision observed", role="bookkeeping"),
    _bin("M_has_wk52", "Week-52 vision observed", role="bookkeeping"),
)

# --------------------------------------------------------------- visit table
VISIT_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("pt_id", "int", "id", "DRCR patient identifier (PtID)", min=1),
    ColumnSpec("treatment", "str", "treatment", "Randomized arm", categories=ARMS),
    ColumnSpec("days_from_rand", "int", "meta", "Days from randomization", min=0, max=1000),
    ColumnSpec("visit_label", "str", "meta", "Nominal visit (baseline, wk4, ...)", True),
    ColumnSpec("va_letters", "float", "outcome", "Study-eye vision (ETDRS letters)", True, 0, 100),
    ColumnSpec(
        "cst_um",
        "float",
        "outcome",
        "Central subfield thickness, machine units (um)",
        True,
        100,
        1200,
    ),
    ColumnSpec(
        "oct_machine",
        "str",
        "meta",
        "OCT machine for this measurement",
        True,
        categories=("Heidelberg Spectralis", "Zeiss Cirrus", "Zeiss Stratus"),
    ),
    _bin("injected", "Study-eye anti-VEGF injection at this visit", role="outcome"),
    _bin("laser", "Study-eye laser at this visit", role="outcome"),
)

ANALYSIS_SPEC: dict[str, ColumnSpec] = {c.name: c for c in ANALYSIS_COLUMNS}
VISIT_SPEC: dict[str, ColumnSpec] = {c.name: c for c in VISIT_COLUMNS}


def feature_columns(columns: Iterable[str] | None = None) -> list[str]:
    """Baseline feature names (``X_`` prefix), in spec order or filtered from ``columns``."""
    names = [c.name for c in ANALYSIS_COLUMNS] if columns is None else list(columns)
    return [c for c in names if c.startswith("X_")]


# ----------------------------------------------------------------- checkers
def _check_columns(
    df: pd.DataFrame, spec: dict[str, ColumnSpec], errors: list[str], allow_extra: bool
) -> None:
    missing = [c for c in spec if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    extra = [c for c in df.columns if c not in spec]
    if extra and not allow_extra:
        errors.append(f"unexpected columns: {extra}")


def _check_column(s: pd.Series, cs: ColumnSpec, errors: list[str]) -> None:
    n_null = int(s.isna().sum())
    if n_null and not cs.nullable:
        errors.append(f"{cs.name}: {n_null} missing values but column is not nullable")
    vals = s.dropna()

    if cs.kind in ("int", "float"):
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            errors.append(f"{cs.name}: expected numeric dtype, got {s.dtype}")
            return
        if cs.kind == "int" and not pd.api.types.is_integer_dtype(s):
            errors.append(f"{cs.name}: expected integer dtype, got {s.dtype}")
        if cs.kind == "float" and len(vals) and not np.all(np.isfinite(vals.to_numpy(float))):
            errors.append(f"{cs.name}: non-finite values")
        if cs.min is not None and (vals < cs.min).any():
            errors.append(f"{cs.name}: {int((vals < cs.min).sum())} values below {cs.min}")
        if cs.max is not None and (vals > cs.max).any():
            errors.append(f"{cs.name}: {int((vals > cs.max).sum())} values above {cs.max}")
    else:
        if pd.api.types.is_numeric_dtype(s) and len(vals):
            errors.append(f"{cs.name}: expected string dtype, got {s.dtype}")
            return

    if cs.categories is not None and len(vals):
        bad = sorted(set(vals.tolist()) - set(cs.categories), key=str)
        if bad:
            errors.append(f"{cs.name}: unexpected levels {bad[:10]}")


def _raise(errors: list[str], table: str) -> None:
    if errors:
        msg = "\n  - ".join(errors)
        raise SchemaError(f"{table} table failed validation ({len(errors)} problems):\n  - {msg}")


def validate_analysis(
    df: pd.DataFrame,
    *,
    expect_trial: bool = False,
    trial: dict[str, Any] | None = None,
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Validate the one-row-per-patient analysis table.

    With ``expect_trial=True`` (real data) also checks the Protocol T facts
    from ``config/default.yaml``: 660 patients, arm sizes 224/218/218 and 44
    patients without week-52 vision. Returns ``df`` unchanged on success.
    """
    errors: list[str] = []
    _check_columns(df, ANALYSIS_SPEC, errors, allow_extra)
    for name, cs in ANALYSIS_SPEC.items():
        if name in df.columns:
            _check_column(df[name], cs, errors)
    _raise(errors, "analysis")  # structural problems first; consistency checks need them

    if df["pt_id"].duplicated().any():
        errors.append(f"pt_id: {int(df['pt_id'].duplicated().sum())} duplicated ids")
    for wk in ("52", "104"):
        has, y = df[f"M_has_wk{wk}"], df[f"Y_va_change_wk{wk}"]
        if not (has.astype(bool) == y.notna()).all():
            errors.append(f"M_has_wk{wk} disagrees with Y_va_change_wk{wk} missingness")
    if (df["Y_cst_change_wk52"].notna() & (df["M_has_wk52"] == 0)).any():
        errors.append("Y_cst_change_wk52 present for patients without a week-52 visit")
    if (df["Y_n_inj_total"] < df["Y_n_inj_yr1"]).any():
        errors.append("Y_n_inj_total < Y_n_inj_yr1 for some patients")
    if (df["Y_laser_any"] < df["Y_laser_yr1"]).any():
        errors.append("Y_laser_any < Y_laser_yr1 for some patients")

    trial = trial if trial is not None else _trial_cfg()
    cutoff = trial["va_below_69_cutoff"]
    if not (df["X_va_below_69"] == (df["X_va_letters"] <= cutoff).astype(int)).all():
        errors.append(f"X_va_below_69 is not (X_va_letters <= {cutoff})")

    if expect_trial:
        if len(df) != trial["n_patients"]:
            errors.append(f"expected {trial['n_patients']} patients, found {len(df)}")
        sizes = df["treatment"].value_counts().to_dict()
        if sizes != trial["arm_sizes"]:
            errors.append(f"arm sizes {sizes} != expected {trial['arm_sizes']}")
        n_miss = int((df["M_has_wk52"] == 0).sum())
        if n_miss != trial["n_missing_wk52"]:
            errors.append(f"{n_miss} patients missing week 52, expected {trial['n_missing_wk52']}")
    _raise(errors, "analysis")
    return df


def validate_visits(
    df: pd.DataFrame,
    *,
    analysis: pd.DataFrame | None = None,
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Validate the long patient x visit table (study eye only).

    If ``analysis`` is given, also checks that every visit patient exists in
    the analysis table with the same arm, and that day 0 vision equals
    ``X_va_letters``.
    """
    errors: list[str] = []
    _check_columns(df, VISIT_SPEC, errors, allow_extra)
    for name, cs in VISIT_SPEC.items():
        if name in df.columns:
            _check_column(df[name], cs, errors)
    _raise(errors, "visits")

    dup = df.duplicated(["pt_id", "days_from_rand"])
    if dup.any():
        errors.append(f"{int(dup.sum())} duplicated (pt_id, days_from_rand) rows")
    if analysis is not None:
        arm = analysis.set_index("pt_id")["treatment"]
        unknown = set(df["pt_id"]) - set(arm.index)
        if unknown:
            errors.append(f"{len(unknown)} visit patients not in the analysis table")
        else:
            mismatch = df["treatment"].to_numpy() != arm.loc[df["pt_id"]].to_numpy()
            if mismatch.any():
                errors.append(f"{int(mismatch.sum())} visit rows disagree with the patient's arm")
            base = df[df["days_from_rand"] == 0].set_index("pt_id")["va_letters"]
            ref = analysis.set_index("pt_id").loc[base.index, "X_va_letters"]
            if not np.allclose(base.to_numpy(float), ref.to_numpy(float), equal_nan=True):
                errors.append("day-0 va_letters differs from X_va_letters")
    _raise(errors, "visits")
    return df


def coerce_dtypes(df: pd.DataFrame, spec: dict[str, ColumnSpec]) -> pd.DataFrame:
    """Cast columns to their spec kind after reading a CSV (ints stay int64)."""
    out = df.copy()
    for name, cs in spec.items():
        if name not in out.columns:
            continue
        if cs.kind == "int" and not out[name].isna().any():
            out[name] = out[name].astype("int64")
        elif cs.kind == "float":
            out[name] = out[name].astype("float64")
        elif cs.kind == "str" and out[name].isna().all():
            out[name] = out[name].astype(object)  # an all-missing text column reads as float
    return out


def _trial_cfg() -> dict[str, Any]:
    from equipoise.config import load_config

    return load_config()["trial"]


def column_table(spec: Sequence[ColumnSpec] = ANALYSIS_COLUMNS) -> pd.DataFrame:
    """Feature description table (Feature | Data Type | Description | Role) for the report."""
    return pd.DataFrame(
        {
            "Feature": [c.name for c in spec],
            "Data Type": [c.kind for c in spec],
            "Description": [c.description for c in spec],
            "Role": [c.role for c in spec],
        }
    )

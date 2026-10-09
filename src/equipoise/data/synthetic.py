"""Synthetic Protocol T-like data with a planted, known treatment effect.

Fake patients are drawn from simple parametric distributions
(``config/synthetic.yaml``) so the tables have exactly the real schema and
realistic ranges, but contain no real patient. Arms are assigned at random.
The week-52 vision change has a known heterogeneous effect: aflibercept helps
more when ``X_va_below_69 = 1``. The noiseless potential-outcome means are
returned as a separate *truth* table so tests can check that methods recover
the planted effect.

Visit-level rows are generated first and the outcome columns (injection
counts, laser flags, vision change) are consistent with them, so the visit
table and the analysis table agree exactly, as the real ones must.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from equipoise.data.schema import ANALYSIS_COLUMNS, VISIT_COLUMNS

DAYS_PER_WEEK = 7
YEAR_WEEKS = 52


@dataclass
class SyntheticData:
    """The three synthetic tables."""

    analysis: pd.DataFrame
    visits: pd.DataFrame
    truth: pd.DataFrame


# ------------------------------------------------------------------ helpers
def _clip_normal(rng: np.random.Generator, spec: dict[str, float], n: int) -> np.ndarray:
    return np.clip(rng.normal(spec["mean"], spec["sd"], n), spec["min"], spec["max"])


def _choice(rng: np.random.Generator, probs: dict[str, float], n: int) -> np.ndarray:
    labels = list(probs)
    p = np.array([probs[k] for k in labels], dtype=float)
    return rng.choice(np.array(labels, dtype=object), size=n, p=p / p.sum())


def _bern(rng: np.random.Generator, p: float | np.ndarray, n: int) -> np.ndarray:
    return (rng.random(n) < p).astype(int)


def _arm_counts(n: int, probs: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding of ``n * p`` so counts sum to ``n``."""
    raw = {a: n * p / sum(probs.values()) for a, p in probs.items()}
    counts = {a: int(np.floor(v)) for a, v in raw.items()}
    for a in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[
        : n - sum(counts.values())
    ]:
        counts[a] += 1
    return counts


# ----------------------------------------------------------------- baseline
def _baseline(rng: np.random.Generator, p: dict[str, Any], n: int, cutoff: int) -> pd.DataFrame:
    b = p["baseline"]
    counts = _arm_counts(n, p["arm_probs"])
    treatment = rng.permutation(np.concatenate([[a] * c for a, c in counts.items()]))
    df = pd.DataFrame({"pt_id": np.arange(1, n + 1), "treatment": treatment.astype(object)})
    df["study_eye"] = rng.choice(np.array(["OD", "OS"], dtype=object), n)

    df["X_age"] = np.round(_clip_normal(rng, b["age"], n)).astype(int)
    df["X_female"] = _bern(rng, b["female_p"], n)
    df["X_hispanic"] = _bern(rng, b["hispanic_p"], n)
    df["X_race"] = _choice(rng, b["race_probs"], n)

    df["X_diabetes_type"] = _choice(rng, b["diabetes_type_probs"], n)
    g = b["diabetes_duration_gamma"]
    df["X_diabetes_duration_yrs"] = np.minimum(
        np.round(rng.gamma(g["shape"], g["scale"], n)), g["max"]
    ).astype(int)
    df["X_insulin"] = _bern(rng, df["X_diabetes_type"].map(b["insulin_p"]).to_numpy(float), n)
    h = b["hba1c_lognormal"]
    df["X_hba1c"] = np.round(
        np.clip(np.exp(rng.normal(h["mean_log"], h["sd_log"], n)), h["min"], h["max"]), 1
    )

    df["X_bmi"] = np.round(_clip_normal(rng, b["bmi"], n), 2)
    sbp = _clip_normal(rng, b["sbp"], n)
    d = b["dbp_given_sbp"]
    dbp = np.clip(d["intercept"] + d["slope"] * sbp + rng.normal(0, d["sd"], n), d["min"], d["max"])
    df["X_sbp"], df["X_dbp"] = np.round(sbp, 1), np.round(dbp, 1)
    for col in (
        "smoker_current",
        "hypertension",
        "prior_mi",
        "cad",
        "prior_stroke",
        "renal_disease",
    ):
        df[f"X_{col}"] = _bern(rng, b[f"{col}_p"], n)
    df["X_log_uacr"] = np.round(_clip_normal(rng, b["log_uacr"], n), 3)

    v = b["va_letters_beta"]
    va = np.round(v["low"] + v["width"] * rng.beta(v["a"], v["b"], n))
    df["X_va_letters"] = va.astype(float)
    df["X_va_below_69"] = (va <= cutoff).astype(int)
    df["X_fellow_va_letters"] = np.round(_clip_normal(rng, b["fellow_va_letters"], n))

    df["X_oct_machine"] = _choice(rng, b["oct_machine_probs"], n)
    c = b["cst_latent_lognormal"]
    latent = np.exp(rng.normal(c["mean_log"], c["sd_log"], n))
    offset = df["X_oct_machine"].map(b["cst_machine_offset_um"]).to_numpy(float)
    rng_c = b["cst_um_range"]
    df["X_cst_um"] = np.round(np.clip(latent + offset, rng_c["min"], rng_c["max"]))
    df["X_subretinal_fluid"] = _bern(rng, b["subretinal_fluid_p"], n).astype(float)
    df["X_epiretinal_membrane"] = _bern(rng, b["epiretinal_membrane_p"], n).astype(float)
    df["X_iop"] = np.round(_clip_normal(rng, b["iop"], n))
    df["X_pseudophakic"] = _bern(rng, b["pseudophakic_p"], n)

    df["X_dr_severity_clinical"] = _choice(rng, b["dr_severity_clinical_probs"], n)
    centre = df["X_dr_severity_clinical"].map(b["dr_severity_rc_centre"]).to_numpy(float)
    df["X_dr_severity_rc"] = np.clip(
        np.round(centre + rng.normal(0, b["dr_severity_rc_sd"], n)), 1, 10
    )

    trt = _bern(rng, b["prior_dme_trt_p"], n)
    df["X_prior_dme_trt"] = trt
    df["X_prior_focal_laser"] = trt * _bern(rng, b["prior_focal_laser_given_trt_p"], n)
    df["X_prior_antivegf"] = trt * _bern(rng, b["prior_antivegf_given_trt_p"], n)
    df["X_prior_prp"] = _bern(rng, b["prior_prp_p"], n)
    return df


# ----------------------------------------------------------------- outcomes
def _potential_outcomes(df: pd.DataFrame, p: dict[str, Any], ref: str) -> pd.DataFrame:
    """Noiseless week-52 mean under the reference arm and the planted effects."""
    o = p["outcome_wk52"]
    cst = df["X_cst_um"].to_numpy(float)
    cst_z = (cst - cst.mean()) / cst.std()
    below = df["X_va_below_69"].to_numpy(float)
    mu_ref = (
        o["base_gain"]
        + o["va_slope"] * (o["va_centre"] - df["X_va_letters"])
        + (o["cst_slope"] * cst_z)
    )
    return pd.DataFrame(
        {
            "pt_id": df["pt_id"],
            "treatment": df["treatment"],
            f"mu_{ref}": np.round(mu_ref, 4),
            "tau_Aflibercept": o["afl_main"] + o["afl_extra_if_va_below_69"] * below,
            "tau_Ranibizumab": o["ran_main"] + o["ran_extra_if_va_below_69"] * below,
        }
    )


def _schedule(rng: np.random.Generator, f: dict[str, Any]) -> list[int]:
    weeks = list(f["year1_visit_weeks"])
    w = weeks[-1]
    while True:
        w += int(rng.choice(f["year2_interval_weeks"], p=f["year2_interval_probs"]))
        if w >= f["year2_end_week"]:
            weeks.append(f["year2_end_week"])
            return weeks
        weeks.append(w)


def generate(params: dict[str, Any], *, va_cutoff: int, reference_arm: str) -> SyntheticData:
    """Generate the analysis, visit and truth tables from ``config/synthetic.yaml`` params."""
    rng = np.random.default_rng(params["seed"])
    n = int(params["n_patients"])
    df = _baseline(rng, params, n, va_cutoff)
    truth = _potential_outcomes(df, params, reference_arm)

    o, o2, cc = params["outcome_wk52"], params["outcome_wk104"], params["cst_change"]
    f, inj, las = params["followup"], params["injections"], params["laser"]
    labels = params["final_status_labels"]
    va0 = df["X_va_letters"].to_numpy(float)
    cst0 = df["X_cst_um"].to_numpy(float)
    arm = df["treatment"].to_numpy()

    tau = np.select(
        [arm == "Aflibercept", arm == "Ranibizumab"],
        [truth["tau_Aflibercept"], truth["tau_Ranibizumab"]],
        0.0,
    )
    y52 = np.round(
        np.clip(va0 + truth[f"mu_{reference_arm}"] + tau + rng.normal(0, o["noise_sd"], n), 0, 100)
        - va0
    )
    y104 = np.round(
        np.clip(va0 + y52 + rng.normal(o2["drift_mean"], o2["drift_sd"], n), 0, 100) - va0
    )
    arm_cst = pd.Series(arm).map(cc["arm_extra_um"]).to_numpy(float)
    floor = 150 - cst0
    c52 = np.maximum(
        np.round(
            -cc["frac_of_excess"] * (cst0 - cc["normal_cst_um"])
            + arm_cst
            + rng.normal(0, cc["noise_sd"], n)
        ),
        floor,
    )
    c104 = np.maximum(
        np.round(c52 + rng.normal(cc["wk104_extra_mean"], cc["wk104_extra_sd"], n)), floor
    )

    u = rng.random(n)
    status = np.where(
        u < f["drop_pre_outcome_p"],
        labels["drop_pre"],
        np.where(
            u < f["drop_pre_outcome_p"] + f["drop_post_outcome_p"],
            labels["drop_post"],
            labels["completed"],
        ),
    ).astype(object)
    missed52 = _bern(rng, f["missed_wk52_visit_p"], n).astype(bool)
    missed104 = _bern(rng, f["missed_wk104_visit_p"], n).astype(bool)

    rows: list[dict[str, Any]] = []
    n_yr1, n_tot = np.zeros(n, int), np.zeros(n, int)
    laser_yr1, laser_any = np.zeros(n, int), np.zeros(n, int)
    has52, has104 = np.zeros(n, int), np.zeros(n, int)
    tau_w = f["trajectory_tau_weeks"]
    shape52 = 1 - np.exp(-YEAR_WEEKS / tau_w)

    for i in range(n):
        weeks = _schedule(rng, f)
        if status[i] == labels["drop_pre"]:
            last = int(rng.choice([w for w in weeks if 0 < w < YEAR_WEEKS]))
            weeks = [w for w in weeks if w <= last]
        elif status[i] == labels["drop_post"]:
            last = int(rng.choice([w for w in weeks if YEAR_WEEKS <= w < f["year2_end_week"]]))
            weeks = [w for w in weeks if w <= last]
        if missed52[i]:
            weeks = [w for w in weeks if w != YEAR_WEEKS]
        if missed104[i]:
            weeks = [w for w in weeks if w != f["year2_end_week"]]
        has52[i] = int(YEAR_WEEKS in weeks)
        has104[i] = int(f["year2_end_week"] in weeks)

        # injections: loading doses then random visits until the year-1 target
        yr1 = [w for w in weeks if w < YEAR_WEEKS]
        target = int(
            np.clip(
                np.round(rng.normal(inj["year1_mean"][arm[i]], inj["year1_sd"])),
                min(inj["loading_doses"], len(yr1)),
                len(yr1),
            )
        )
        injected = set(yr1[: inj["loading_doses"]])
        rest = yr1[inj["loading_doses"] :]
        extra = max(0, target - len(injected))
        injected |= set(rng.choice(rest, size=extra, replace=False).tolist()) if extra else set()
        for w in weeks:
            if (
                YEAR_WEEKS <= w < f["year2_end_week"]
                and rng.random() < inj["year2_inject_p"][arm[i]]
            ):
                injected.add(w)

        lasered: set[int] = set()
        elig1 = [w for w in yr1 if w >= las["first_allowed_week"]]
        if elig1 and rng.random() < las["year1_p"][arm[i]]:
            lasered.add(int(rng.choice(elig1)))
        elig2 = [w for w in weeks if YEAR_WEEKS <= w < f["year2_end_week"]]
        if elig2 and rng.random() < las["year2_new_p"][arm[i]]:
            lasered.add(int(rng.choice(elig2)))

        n_yr1[i] = sum(w < YEAR_WEEKS for w in injected)
        n_tot[i] = len(injected)
        laser_yr1[i] = int(any(w < YEAR_WEEKS for w in lasered))
        laser_any[i] = int(bool(lasered))

        machine = df.at[i, "X_oct_machine"]
        for w in weeks:
            if w == 0:
                day, va, cst = 0, va0[i], cst0[i]
            else:
                day = w * DAYS_PER_WEEK + int(
                    rng.integers(-f["visit_jitter_days"], f["visit_jitter_days"] + 1)
                )
                if w <= YEAR_WEEKS:
                    frac = (1 - np.exp(-w / tau_w)) / shape52
                    va_mean, cst_mean = va0[i] + y52[i] * frac, cst0[i] + c52[i] * frac
                else:
                    frac = (w - YEAR_WEEKS) / (f["year2_end_week"] - YEAR_WEEKS)
                    va_mean = va0[i] + y52[i] + (y104[i] - y52[i]) * frac
                    cst_mean = cst0[i] + c52[i] + (c104[i] - c52[i]) * frac
                exact = w in (YEAR_WEEKS, f["year2_end_week"])
                va = va_mean if exact else va_mean + rng.normal(0, f["va_visit_noise_sd"])
                va = float(np.clip(np.round(va), 0, 100))
                if exact:
                    cst = cst_mean
                elif rng.random() < f["oct_measured_p"]:
                    cst = max(150.0, cst_mean + rng.normal(0, f["cst_visit_noise_sd"]))
                else:
                    cst = np.nan
                cst = float(np.round(cst)) if np.isfinite(cst) else np.nan
            rows.append(
                {
                    "pt_id": int(df.at[i, "pt_id"]),
                    "treatment": arm[i],
                    "days_from_rand": day,
                    "visit_label": "baseline" if w == 0 else f"wk{w}",
                    "va_letters": va,
                    "cst_um": cst,
                    "oct_machine": machine,
                    "injected": int(w in injected),
                    "laser": int(w in lasered),
                }
            )

    df["Y_va_change_wk52"] = np.where(has52 == 1, y52, np.nan)
    df["Y_va_change_wk104"] = np.where(has104 == 1, y104, np.nan)
    extra_miss = _bern(rng, cc["extra_missing_p"], n).astype(bool)
    df["Y_cst_change_wk52"] = np.where((has52 == 1) & ~extra_miss, c52, np.nan)
    df["Y_cst_change_wk104"] = np.where(has104 == 1, c104, np.nan)
    df["Y_n_inj_yr1"], df["Y_n_inj_total"] = n_yr1, n_tot
    df["Y_laser_yr1"], df["Y_laser_any"] = laser_yr1, laser_any
    df["M_final_status"] = status
    df["M_has_wk52"], df["M_has_wk104"] = has52, has104

    visits = pd.DataFrame(rows)
    # the week-52 CST that went "missing" in the analysis table is missing in the visits too
    miss_ids = set(df.loc[extra_miss & (has52 == 1), "pt_id"])
    wk52 = visits["visit_label"].eq(f"wk{YEAR_WEEKS}") & visits["pt_id"].isin(miss_ids)
    visits.loc[wk52, "cst_um"] = np.nan

    df = _apply_missing(rng, df, params["missing_rates"])
    # outcome CST change needs a baseline CST; visits mirror the baseline gaps
    no_cst = df["X_cst_um"].isna()
    df.loc[no_cst, ["Y_cst_change_wk52", "Y_cst_change_wk104"]] = np.nan
    base_rows = visits["days_from_rand"].eq(0)
    visits.loc[base_rows, "cst_um"] = (
        visits.loc[base_rows, "pt_id"].map(df.set_index("pt_id")["X_cst_um"]).to_numpy()
    )
    visits["oct_machine"] = visits["pt_id"].map(df.set_index("pt_id")["X_oct_machine"])
    visits.loc[visits["cst_um"].isna(), "oct_machine"] = np.nan

    analysis = df[[c.name for c in ANALYSIS_COLUMNS]]
    visits = visits[[c.name for c in VISIT_COLUMNS]].sort_values(["pt_id", "days_from_rand"])
    return SyntheticData(analysis.reset_index(drop=True), visits.reset_index(drop=True), truth)


def _apply_missing(
    rng: np.random.Generator, df: pd.DataFrame, rates: dict[str, float]
) -> pd.DataFrame:
    out = df.copy()
    for col, rate in rates.items():
        mask = rng.random(len(out)) < rate
        if out[col].dtype.kind in "iu":
            out[col] = out[col].astype(float)
        out.loc[mask, col] = np.nan
    return out

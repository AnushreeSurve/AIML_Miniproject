"""Shared preprocessor for every model: impute, transform, encode, scale.

``build_preprocessor(cfg)`` returns one sklearn ``ColumnTransformer`` that
all later models put at the front of their ``Pipeline``, so every step is
fitted on the training fold only. Continuous features are log-transformed
where skewed (HbA1c), standardized and imputed by k-nearest neighbours;
binary features are imputed with the most frequent value; nominal features
are one-hot encoded (with a "Missing" level); DR severity is ordinal-encoded
in its clinical order; and CST is harmonized within OCT machine. Only ``X_``
columns are selected, so outcomes can never leak in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from equipoise.config import load_config
from equipoise.data.features import assert_no_leakage, register_feature_builder
from equipoise.preprocess.harmonize import MachineZScore

MISSING_LEVEL = "Missing"


def _numeric_imputer(pcfg: dict[str, Any], seed: int):
    kind = pcfg["imputer"]
    if kind == "knn":
        return KNNImputer(n_neighbors=pcfg["knn_neighbors"])
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "iterative":
        return IterativeImputer(random_state=seed, max_iter=pcfg["missing"]["mice_max_iter"])
    raise ValueError(f"unknown preprocess.imputer: {kind}")


def _log_columns(X: np.ndarray, flags: np.ndarray) -> np.ndarray:
    out = np.array(X, dtype=float, copy=True)
    out[:, flags] = np.log(out[:, flags])
    return out


def _to_object(X):
    """Cast to object so string and missing values can share one imputer."""
    return pd.DataFrame(X).astype(object).where(pd.DataFrame(X).notna(), np.nan)


def build_preprocessor(cfg: dict[str, Any] | None = None) -> ColumnTransformer:
    """Return the unfitted ``ColumnTransformer`` used by all baseline models."""
    cfg = cfg or load_config()
    f, pcfg = cfg["features"], cfg["preprocess"]
    cont = list(f["continuous"])
    log_flags = np.array([c in f["log_transform"] for c in cont])
    ordinal_cols = list(f["ordinal"])

    continuous = Pipeline(
        [
            (
                "log",
                FunctionTransformer(
                    _log_columns, kw_args={"flags": log_flags}, feature_names_out="one-to-one"
                ),
            ),
            ("scale", StandardScaler()),  # ignores NaN when fitting, keeps them
            ("impute", _numeric_imputer(pcfg, cfg["seed"])),
        ]
    )
    cst = Pipeline(
        [
            ("harmonize", MachineZScore(min_group_n=pcfg["cst_min_machine_n"])),
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    nominal = Pipeline(
        [
            ("as_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_LEVEL)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    ordinal = Pipeline(
        [
            ("as_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=[f["ordinal"][c] for c in ordinal_cols])),
            ("scale", StandardScaler()),
        ]
    )
    ct = ColumnTransformer(
        [
            ("num", continuous, cont),
            ("cst", cst, [f["cst"]["value"], f["cst"]["machine"]]),
            ("bin", binary, list(f["binary"])),
            ("cat", nominal, list(f["nominal"])),
            ("ord", ordinal, ordinal_cols),
        ],
        remainder="drop",
        sparse_threshold=pcfg["sparse_threshold"],
        verbose_feature_names_out=True,
    )
    used = (
        cont
        + [f["cst"]["value"], f["cst"]["machine"]]
        + list(f["binary"])
        + list(f["nominal"])
        + ordinal_cols
    )
    assert_no_leakage(used)
    return ct


def transform_frame(ct: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a *fitted* preprocessor and return a named DataFrame."""
    Z = ct.transform(df)
    Z = Z.toarray() if sp.issparse(Z) else Z
    return pd.DataFrame(Z, columns=ct.get_feature_names_out(), index=df.index)


@register_feature_builder("preprocessed_features")
def preprocessed_features(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Fit the preprocessor on ``df`` and return the encoded baseline matrix (leakage-checked)."""
    ct = build_preprocessor(cfg).fit(df)
    out = transform_frame(ct, df)
    assert_no_leakage(out.columns)
    return out


def onehot_sparse(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> tuple[sp.csr_matrix, list]:
    """One-hot encode the nominal features as a ``scipy.sparse`` CSR matrix.

    Most entries of a one-hot matrix are zero, so CSR stores only the
    non-zeros (see ``docs/data_structures.md``).
    """
    cfg = cfg or load_config()
    cols = list(cfg["features"]["nominal"])
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X = df[cols].astype(object).where(df[cols].notna(), MISSING_LEVEL)
    M = enc.fit_transform(X).tocsr()
    return M, list(enc.get_feature_names_out(cols))

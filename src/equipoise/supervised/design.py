"""Design matrices, CV folds and naive baselines shared by the M5 outcome models.

Every model is a sklearn ``Pipeline``: the shared baseline preprocessor plus a
one-hot encoding of the randomized arm (bevacizumab as the dropped reference)
followed by the estimator, so all preprocessing is refitted inside each CV
fold. Cross-validation reuses the fixed folds saved in ``splits.json``.
Baselines predict the training mean (or event rate) of each arm, which is
what a model that ignores every baseline feature would do.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from equipoise.data.features import assert_no_leakage
from equipoise.data.splits import Splits
from equipoise.preprocess.encode import build_preprocessor


def x_columns(df: pd.DataFrame) -> list[str]:
    """Baseline feature columns present in ``df``."""
    return [c for c in df.columns if c.startswith("X_")]


def arm_encoder(cfg: dict[str, Any]) -> OneHotEncoder:
    """One-hot encoder for ``treatment`` with the reference arm dropped."""
    arms = cfg["analysis"]["arms"]
    return OneHotEncoder(
        categories=[arms], drop=[cfg["analysis"]["reference_arm"]], sparse_output=False
    )


def outcome_preprocessor(cfg: dict[str, Any], df: pd.DataFrame) -> ColumnTransformer:
    """Baseline preprocessor + arm one-hot (the arm is an explicit model input in M5)."""
    xs = x_columns(df)
    assert_no_leakage([*xs, "treatment"], allow_treatment=True)
    return ColumnTransformer(
        [("base", build_preprocessor(cfg), xs), ("arm", arm_encoder(cfg), ["treatment"])],
        remainder="drop",
    )


def full_rank_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    """Scaled continuous + binary + arm dummies, no one-hot nominal: invertible X^T X.

    Used only to verify the from-scratch solvers (normal equations, IRLS)
    against library implementations on an identical, full-rank design.
    """
    from sklearn.impute import SimpleImputer

    f = cfg["features"]
    num = [*f["continuous"], f["cst"]["value"]]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                num,
            ),
            (
                "bin",
                SimpleImputer(strategy="most_frequent"),
                [c for c in f["binary"] if c != "X_va_below_69"],
            ),
            ("arm", arm_encoder(cfg), ["treatment"]),
        ],
        remainder="drop",
    )


def predefined_cv(rows: pd.DataFrame, splits: Splits) -> PredefinedSplit:
    """``PredefinedSplit`` using each training patient's saved fold."""
    fold = dict(zip(splits.train_ids, splits.train_fold, strict=True))
    missing = set(rows["pt_id"]) - set(fold)
    if missing:
        raise ValueError(f"{len(missing)} rows are not training patients (test leakage?)")
    return PredefinedSplit(rows["pt_id"].map(fold).to_numpy())


class ArmMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the training mean outcome of the patient's arm."""

    def fit(self, X: pd.DataFrame, y) -> ArmMeanRegressor:
        """Store per-arm means (and the overall mean for unseen arms)."""
        y = pd.Series(np.asarray(y, float), index=X.index)
        self.means_ = y.groupby(X["treatment"]).mean().to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Per-arm training mean."""
        return X["treatment"].map(self.means_).fillna(self.overall_).to_numpy(float)


class ArmRateClassifier(ClassifierMixin, BaseEstimator):
    """Predict the training event rate of the patient's arm."""

    def fit(self, X: pd.DataFrame, y) -> ArmRateClassifier:
        """Store per-arm event rates."""
        y = pd.Series(np.asarray(y, int), index=X.index)
        self.classes_ = np.array([0, 1])
        self.rates_ = y.groupby(X["treatment"]).mean().to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Two-column probabilities from the per-arm rate."""
        p = X["treatment"].map(self.rates_).fillna(self.overall_).to_numpy(float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Majority class within the arm (threshold 0.5)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

"""Model zoo for M5, built from config, and the tune-then-evaluate loop.

Each model is ``Pipeline([preprocessor, estimator])`` tuned by grid search
over the config grid using the saved 5 training folds. The reported CV
metrics come from those same folds for the chosen hyper-parameters (so they
are slightly optimistic); the held-out test metrics are the honest ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from equipoise.supervised.scratch import IRLSLogisticRegression

REG_LABELS = {
    "ols": "OLS (least squares)",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "knn": "kNN",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def regression_models(cfg: dict[str, Any]) -> dict[str, tuple[BaseEstimator, dict]]:
    """Name -> (estimator, grid) for the vision-change regression."""
    g, seed = cfg["supervised"]["regression"], cfg["seed"]
    est = {
        "ols": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=20000),
        "knn": KNeighborsRegressor(),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }
    return {k: (est[k], g[k]) for k in g}


def classification_models(
    cfg: dict[str, Any], class_weight: str | None
) -> dict[str, tuple[BaseEstimator, dict]]:
    """Name -> (estimator, grid) for the binary targets."""
    g, seed, ir = cfg["supervised"]["classification"], cfg["seed"], cfg["supervised"]["irls"]
    out = {
        "logistic": (LogisticRegression(max_iter=5000, class_weight=class_weight), g["logistic"]),
        "logistic_irls_scratch": (
            IRLSLogisticRegression(ir["max_iter"], ir["tol"], ir["l2"], class_weight),
            {},
        ),
        "knn": (KNeighborsClassifier(), g["knn"]),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed, class_weight=class_weight),
            g["decision_tree"],
        ),
        "gaussian_nb": (GaussianNB(), g["gaussian_nb"]),
        "bernoulli_nb": (BernoulliNB(binarize=0.0), g["bernoulli_nb"]),
    }
    return out


def count_models(cfg: dict[str, Any]) -> dict[str, tuple[BaseEstimator, dict]]:
    """Name -> (estimator, grid) for year-1 injection counts."""
    g = cfg["supervised"]["count"]
    return {
        "poisson_glm": (PoissonRegressor(max_iter=1000), g["poisson_glm"]),
        "gradient_boosting_poisson": (
            HistGradientBoostingRegressor(loss="poisson", random_state=cfg["seed"]),
            g["gradient_boosting_poisson"],
        ),
    }


@dataclass
class Fitted:
    """A tuned pipeline with its best parameters and CV score."""

    name: str
    model: Any
    best_params: dict
    cv_results: dict


def tune(
    name: str,
    est: BaseEstimator,
    grid: dict,
    pre,
    X: pd.DataFrame,
    y: np.ndarray,
    cv,
    scoring: str,
    extra_scoring: dict,
) -> Fitted:
    """Grid-search ``Pipeline([pre, est])`` on the CV folds; refit on all training rows."""
    pipe = Pipeline([("pre", clone(pre)), ("model", clone(est))])
    param_grid = {f"model__{k}": v for k, v in grid.items()}
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(X, y)
        best, params = (
            gs.best_estimator_,
            {k.removeprefix("model__"): v for k, v in gs.best_params_.items()},
        )
    else:
        best, params = pipe.fit(X, y), {}
    cvr = cross_validate(clone(best), X, y, cv=cv, scoring=extra_scoring, n_jobs=-1)
    return Fitted(name, best, params, cvr)


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """MSE, RMSE, MAE and R^2."""
    mse = mean_squared_error(y, yhat)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y, yhat),
        "R2": r2_score(y, yhat),
    }


def classification_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC and PR-AUC."""
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "Accuracy": (tp + tn) / len(y),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(y, yhat, zero_division=0),
        "AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def count_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and mean Poisson deviance."""
    yhat = np.clip(yhat, 1e-6, None)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "MAE": mean_absolute_error(y, yhat),
        "Poisson_deviance": mean_poisson_deviance(y, yhat),
    }


def summarize_cv(cvr: dict, keys: dict[str, tuple[str, float]]) -> dict[str, float]:
    """Mean CV metrics; ``keys`` maps output name -> (sklearn key, sign)."""
    return {name: float(sign * np.mean(cvr[f"test_{k}"])) for name, (k, sign) in keys.items()}

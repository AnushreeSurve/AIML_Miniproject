"""From-scratch solvers: OLS by the normal equations, logistic regression by IRLS.

*Least squares* minimises ``||y - X b||^2``; setting the gradient to zero
gives the normal equations ``X^T X b = X^T y``, solved here with
``numpy.linalg.solve`` (valid when X has full column rank).
*Logistic regression* has no closed form, so we use *Newton-Raphson*: with
p = sigmoid(X b), the gradient is ``X^T (y - p)`` and the Hessian is
``-X^T W X`` with ``W = diag(p (1 - p))``. Each Newton step solves a
weighted least-squares problem, hence the name *iteratively reweighted
least squares* (IRLS). Both are checked against sklearn / statsmodels.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def normal_equations(X: np.ndarray, y: np.ndarray, intercept: bool = True) -> np.ndarray:
    """Solve ``X^T X b = X^T y`` (an intercept column is prepended if requested)."""
    X = np.asarray(X, float)
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(X.T @ X, X.T @ np.asarray(y, float))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class IRLSLogisticRegression(ClassifierMixin, BaseEstimator):
    """Logistic regression fitted by Newton-Raphson / IRLS.

    ``l2`` adds ``l2 * ||b||^2 / 2`` (intercept excluded) so a rank-deficient
    design, e.g. a full one-hot block, still has an invertible Hessian.
    ``class_weight="balanced"`` reweights observations by inverse class frequency.
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-8,
        l2: float = 0.0,
        class_weight: str | None = None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2
        self.class_weight = class_weight

    def fit(self, X, y) -> IRLSLogisticRegression:
        """Run Newton steps until the coefficient change is below ``tol``."""
        X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        y = np.asarray(y, float)
        self.classes_ = np.array([0, 1])
        w_obs = np.ones(len(y))
        if self.class_weight == "balanced":
            n1 = y.sum()
            w_obs = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * (len(y) - n1)))
        penalty = self.l2 * np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        b = np.zeros(X.shape[1])
        self.loglik_path_ = []
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            self.n_iter_ += 1
            p = _sigmoid(X @ b)
            grad = X.T @ (w_obs * (y - p)) - penalty @ b
            H = X.T @ (X * (w_obs * p * (1 - p))[:, None]) + penalty
            step = np.linalg.solve(H, grad)
            b = b + step
            p = _sigmoid(X @ b)
            eps = 1e-12
            self.loglik_path_.append(
                float(np.sum(w_obs * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
            )
            if np.max(np.abs(step)) < self.tol:
                break
        self.intercept_ = np.array([b[0]])
        self.coef_ = b[1:][None, :]
        return self

    def decision_function(self, X) -> np.ndarray:
        """Linear predictor ``b0 + X b``."""
        return np.asarray(X, float) @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        """Two-column class probabilities."""
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        """Class at threshold 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

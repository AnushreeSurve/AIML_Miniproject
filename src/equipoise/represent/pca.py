"""PCA computed directly from the singular value decomposition.

After centring the feature matrix X (patients x features), the SVD
``X = U S V^T`` gives the principal directions as the rows of ``V^T`` and the
patient coordinates (scores) as ``U S``. The variance explained by component
j is ``s_j^2 / (n - 1)``, so the explained-variance ratio is
``s_j^2 / sum(s^2)``. We implement this with ``numpy.linalg.svd`` and check it
against ``sklearn.decomposition.PCA`` (identical up to the sign of each
component, which is arbitrary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class SVDPCA:
    """Fitted PCA: mean, components (k x p), singular values and variance ratios."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project rows of X onto the components (scores)."""
        return (np.asarray(X, float) - self.mean) @ self.components.T


def svd_pca(X: np.ndarray, n_components: int | None = None) -> SVDPCA:
    """Fit PCA via ``numpy.linalg.svd`` on the centred matrix."""
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    # fix the arbitrary sign: largest-|loading| entry of each component positive
    signs = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    Vt = Vt * signs[:, None]
    var = s**2 / (len(X) - 1)
    k = n_components or len(s)
    return SVDPCA(mean, Vt[:k], s[:k], var[:k], (var / var.sum())[:k])


def compare_with_sklearn(X: np.ndarray, n_components: int) -> dict[str, float]:
    """Max absolute differences between our SVD-PCA and sklearn's PCA."""
    ours = svd_pca(X, n_components)
    sk = PCA(n_components=n_components, svd_solver="full").fit(X)
    sign = np.sign(np.sum(ours.components * sk.components_, axis=1))
    return {
        "max_abs_diff_components": float(
            np.abs(ours.components - sign[:, None] * sk.components_).max()
        ),
        "max_abs_diff_explained_ratio": float(
            np.abs(ours.explained_variance_ratio - sk.explained_variance_ratio_).max()
        ),
        "max_abs_diff_scores": float(np.abs(ours.transform(X) - sign * sk.transform(X)).max()),
    }

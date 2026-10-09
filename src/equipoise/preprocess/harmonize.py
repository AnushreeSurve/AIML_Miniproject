"""CST harmonization across OCT machines (within-machine z-scoring).

Protocol T measured central subfield thickness (CST) on three OCT machines
(Zeiss Stratus, Zeiss Cirrus, Heidelberg Spectralis), and spectral-domain
machines read thicker than time-domain Stratus. Published conversion
equations exist (e.g. the DRCR Retina Network's Stratus-equivalent
conversions, Bressler et al., JAMA Ophthalmology 2014), but we could not
verify the exact coefficients, so we do **not** apply them. Instead each eye's
CST is z-scored *within its own machine*:
``z = (CST - mean_machine) / sd_machine``, with the means and SDs learned on
the training fold only. Machines with too few training eyes (Stratus has
~26) fall back to the pooled mean/SD.

Caveat for the viva: within-machine z-scoring removes any real difference in
thickness between patients measured on different machines, as well as the
device bias. In Protocol T, eligibility thresholds were themselves
machine-specific, so this is a reasonable trade-off here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

POOLED = "__pooled__"


class MachineZScore(TransformerMixin, BaseEstimator):
    """Z-score a measurement within the group given by a second column.

    Input: a 2-column frame ``[value, machine]``. Output: one column
    ``<value>_harm``. Missing values stay missing (imputed later); a missing
    or unseen machine uses the pooled statistics.
    """

    def __init__(self, min_group_n: int = 20):
        self.min_group_n = min_group_n

    def fit(self, X: pd.DataFrame, y=None) -> MachineZScore:
        """Learn per-machine mean and SD on the training rows."""
        X = self._frame(X)
        value, group = X.iloc[:, 0].astype(float), X.iloc[:, 1]
        self.value_name_ = str(X.columns[0])
        self.stats_: dict[str, tuple[float, float, int]] = {
            POOLED: (float(value.mean()), float(value.std(ddof=1)), int(value.notna().sum()))
        }
        for g, v in value.groupby(group, dropna=True):
            n = int(v.notna().sum())
            if n >= self.min_group_n:
                self.stats_[str(g)] = (float(v.mean()), float(v.std(ddof=1)), n)
        self.n_features_in_ = 2
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return the within-machine z-score as an ``(n, 1)`` array."""
        check_is_fitted(self, "stats_")
        X = self._frame(X)
        value, group = X.iloc[:, 0].astype(float).to_numpy(), X.iloc[:, 1]
        keys = [str(g) if pd.notna(g) and str(g) in self.stats_ else POOLED for g in group]
        mean = np.array([self.stats_[k][0] for k in keys])
        sd = np.array([self.stats_[k][1] for k in keys])
        return ((value - mean) / sd).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Single output column ``<value>_harm``."""
        check_is_fitted(self, "stats_")
        return np.array([f"{self.value_name_}_harm"], dtype=object)

    def stats_table(self) -> pd.DataFrame:
        """Fitted mean / SD / n per machine (``pooled`` row included)."""
        check_is_fitted(self, "stats_")
        return pd.DataFrame(
            [
                {"machine": "pooled" if k == POOLED else k, "mean_um": m, "sd_um": s, "n_train": n}
                for k, (m, s, n) in self.stats_.items()
            ]
        )

    @staticmethod
    def _frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=["value", "group"])

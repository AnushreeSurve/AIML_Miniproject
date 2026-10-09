"""Holm's step-down correction for multiple comparisons.

With three pairwise arm comparisons the chance of at least one false
positive at alpha = 0.05 is about 14 %. Holm sorts the p-values, multiplies
the smallest by m, the next by m - 1, and so on, keeping the adjusted values
monotone. It controls the family-wise error rate like Bonferroni but is
uniformly more powerful.
"""

from __future__ import annotations

import numpy as np


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj

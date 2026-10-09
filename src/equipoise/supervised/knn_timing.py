"""kNN neighbour search: KD-tree vs ball tree vs brute force.

Brute force computes all n_query x n_train distances. A *KD-tree* splits
space on one coordinate at a time, and a *ball tree* nests hyperspheres; both
can skip far-away regions, but in tens of dimensions their pruning weakens
(the curse of dimensionality), so the gain over brute force is small or
negative. All three must return the same neighbour distances.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compare(
    Z_train: np.ndarray, Z_query: np.ndarray, algorithms: list[str], k: int, repeats: int
) -> pd.DataFrame:
    """Median fit and query time per algorithm, and agreement with brute force."""
    ref = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(Z_train).kneighbors(Z_query)[0]
    rows = []
    for algo in algorithms:
        fit_t, q_t = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=k, algorithm=algo).fit(Z_train)
            t1 = time.perf_counter()
            dist, _ = nn.kneighbors(Z_query)
            t2 = time.perf_counter()
            fit_t.append(t1 - t0)
            q_t.append(t2 - t1)
        rows.append(
            {
                "algorithm": algo,
                "fit_ms_median": 1e3 * np.median(fit_t),
                "query_ms_median": 1e3 * np.median(q_t),
                "max_abs_distance_diff_vs_brute": float(np.abs(dist - ref).max()),
                "n_train": len(Z_train),
                "n_query": len(Z_query),
                "dims": Z_train.shape[1],
                "k": k,
            }
        )
    return pd.DataFrame(rows)

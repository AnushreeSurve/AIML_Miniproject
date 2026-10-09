"""Outlier detection on baseline features: IQR rule, Isolation Forest, DBSCAN.

Three detectors with different ideas of "unusual". The *IQR rule* flags a
patient if any continuous feature lies more than 1.5 interquartile ranges
outside the middle 50 %. *Isolation Forest* builds random trees and flags
points that are isolated in few splits. *DBSCAN* clusters dense regions and
labels points in no dense region as noise; its ``eps`` is read off the
k-distance curve. Detectors are fitted on the training set; patients are
only *flagged*, never dropped, and the overlap between methods is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from equipoise.preprocess.encode import build_preprocessor, transform_frame


def iqr_flags(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray, k: float) -> pd.DataFrame:
    """Per-feature IQR outlier flags with quartiles from the training rows."""
    flags = pd.DataFrame(index=df.index)
    for c in cols:
        q1, q3 = df.loc[train_mask, c].quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        flags[c] = ((df[c] < lo) | (df[c] > hi)).fillna(False)
    return flags


def k_distance(Z: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance of each point to its k-th nearest neighbour (for choosing eps)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, _ = nn.kneighbors(Z)
    return np.sort(dist[:, k])


def detect(df: pd.DataFrame, train_mask: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the three detectors; return flags, counts, overlaps and the k-distance curve."""
    ocfg = cfg["preprocess"]["outliers"]
    cont = list(cfg["features"]["continuous"]) + [cfg["features"]["cst"]["value"]]
    iqr = iqr_flags(df, cont, train_mask, ocfg["iqr_k"])

    ct = build_preprocessor(cfg).fit(df[train_mask])
    Z_all = transform_frame(ct, df)
    num_cols = [c for c in Z_all.columns if c.startswith(("num__", "cst__"))]
    Z = Z_all[num_cols].to_numpy()
    Z_train = Z[train_mask]

    iso = IsolationForest(
        contamination=ocfg["isolation_contamination"], random_state=cfg["seed"]
    ).fit(Z_train)
    iso_flag = iso.predict(Z) == -1

    kd = k_distance(Z_train, ocfg["dbscan_min_samples"])
    eps = float(np.quantile(kd, ocfg["dbscan_eps_quantile"]))
    db = DBSCAN(eps=eps, min_samples=ocfg["dbscan_min_samples"]).fit(Z_train)
    core = Z_train[db.core_sample_indices_]
    # a point is noise if it is not within eps of any training core point
    d_core, _ = NearestNeighbors(n_neighbors=1).fit(core).kneighbors(Z)
    db_flag = d_core[:, 0] > eps

    flags = pd.DataFrame(
        {
            "pt_id": df["pt_id"].to_numpy(),
            "iqr": iqr.any(axis=1).to_numpy(),
            "isolation_forest": iso_flag,
            "dbscan_noise": db_flag,
        }
    )
    counts = pd.DataFrame(
        {
            "method": ["IQR rule (any feature)", "Isolation Forest", "DBSCAN noise"],
            "n_flagged": [int(flags[c].sum()) for c in ("iqr", "isolation_forest", "dbscan_noise")],
        }
    )
    counts["pct"] = 100 * counts["n_flagged"] / len(df)
    combo = (
        flags[["iqr", "isolation_forest", "dbscan_noise"]]
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )
    names = {
        "000": "none",
        "100": "IQR only",
        "010": "IForest only",
        "001": "DBSCAN only",
        "110": "IQR + IForest",
        "101": "IQR + DBSCAN",
        "011": "IForest + DBSCAN",
        "111": "all three",
    }
    overlap = combo.map(names).value_counts().rename_axis("flagged_by").reset_index(name="n")
    per_feature = pd.DataFrame(
        {"feature": cont, "n_iqr_flagged": [int(iqr[c].sum()) for c in cont]}
    )
    return {
        "flags": flags,
        "counts": counts,
        "overlap": overlap,
        "per_feature": per_feature,
        "k_distance": kd,
        "eps": eps,
        "n_clusters": int(len(set(db.labels_) - {-1})),
    }

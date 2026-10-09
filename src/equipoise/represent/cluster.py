"""Clustering of patients on scaled baseline features (exploratory).

*k-means* places k centroids to minimise within-cluster squared distance; we
plot the elbow (inertia vs k) and pick k with the best *silhouette* (how much
closer a point is to its own cluster than to the next). *Hierarchical Ward*
clustering merges the two clusters whose union increases the total
within-cluster variance least, drawn as a dendrogram. *DBSCAN* groups points
in dense regions and calls the rest noise; ``eps`` is read off the
k-distance plot. Within each cluster we then report the arm differences in
week-52 vision with CIs. With ~220 patients per arm these are exploratory
descriptions, not evidence of subgroup effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from equipoise.preprocess.outliers import k_distance
from equipoise.stats.effects import arm_differences


def kmeans_scan(Z: np.ndarray, k_range: list[int], seed: int, n_init: int) -> pd.DataFrame:
    """Inertia and silhouette for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_)})
    return pd.DataFrame(rows)


def ward_scan(L: np.ndarray, Z: np.ndarray, k_range: list[int]) -> pd.DataFrame:
    """Silhouette of the Ward tree cut into k clusters."""
    return pd.DataFrame(
        [{"k": k, "silhouette": silhouette_score(Z, fcluster(L, k, "maxclust"))} for k in k_range]
    )


def fit_clusterings(Z: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run k-means, Ward and DBSCAN; return labels and the scans used to choose them."""
    r = cfg["represent"]
    km_scan = kmeans_scan(Z, r["k_range"], cfg["seed"], r["kmeans_n_init"])
    k_km = int(km_scan.loc[km_scan.silhouette.idxmax(), "k"])
    km = KMeans(n_clusters=k_km, n_init=r["kmeans_n_init"], random_state=cfg["seed"]).fit(Z)

    L = linkage(Z, method="ward")
    w_scan = ward_scan(L, Z, r["k_range"])
    k_w = int(w_scan.loc[w_scan.silhouette.idxmax(), "k"])
    ward_labels = fcluster(L, k_w, "maxclust") - 1

    kd = k_distance(Z, r["dbscan_min_samples"])
    eps = float(np.quantile(kd, r["dbscan_eps_quantile"]))
    db = DBSCAN(eps=eps, min_samples=r["dbscan_min_samples"]).fit(Z)
    return {
        "kmeans": km.labels_,
        "ward": ward_labels,
        "dbscan": db.labels_,
        "kmeans_scan": km_scan,
        "ward_scan": w_scan,
        "linkage": L,
        "k_distance": kd,
        "eps": eps,
        "k_kmeans": k_km,
        "k_ward": k_w,
    }


def effects_by_cluster(
    df: pd.DataFrame, labels: np.ndarray, method: str, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Arm differences in the primary outcome within each cluster (label -1 = noise)."""
    outcome = cfg["analysis"]["primary_outcome"]
    min_n = cfg["represent"]["min_arm_n_per_cluster"]
    rows = []
    for lab in sorted(set(labels)):
        sub = df[labels == lab]
        name = "noise" if lab == -1 else str(lab)
        for r in arm_differences(sub, outcome, cfg, min_n=min_n):
            rows.append({"method": method, "cluster": name, "n_patients": len(sub), **r})
    out = pd.DataFrame(rows)
    out["note"] = "exploratory - not a confirmed subgroup effect"
    return out

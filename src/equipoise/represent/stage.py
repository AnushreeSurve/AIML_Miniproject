"""M3 stage: PCA/SVD map, clustering with within-cluster effects, association rules.

Everything here is exploratory and uses the training patients only (the
held-out test set is reserved for final reporting of the models).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from equipoise import tracking
from equipoise.pipeline import prepare
from equipoise.preprocess.discretize import discretize
from equipoise.preprocess.encode import build_preprocessor, transform_frame
from equipoise.represent import cluster, rules
from equipoise.represent.pca import compare_with_sklearn, svd_pca

MODULE = "m3"
EXPERIMENT = "m3-represent"
COLORS = {"Aflibercept": "#4C72B0", "Bevacizumab": "#DD8452", "Ranibizumab": "#55A868"}


def _pca_section(Z: pd.DataFrame, train: pd.DataFrame, cfg, rep) -> dict[str, float]:
    k = min(cfg["represent"]["n_components"], Z.shape[1])
    p = svd_pca(Z.to_numpy(), k)
    check = compare_with_sklearn(Z.to_numpy(), k)
    rep.table(pd.DataFrame([check]), "pca_vs_sklearn")
    rep.table(
        pd.DataFrame(
            {
                "component": np.arange(1, k + 1),
                "singular_value": p.singular_values,
                "explained_variance": p.explained_variance,
                "explained_ratio": p.explained_variance_ratio,
                "cumulative": np.cumsum(p.explained_variance_ratio),
            }
        ),
        "pca_scree",
    )
    load = pd.DataFrame(p.components[:3].T, index=Z.columns, columns=["PC1", "PC2", "PC3"])
    top = load.abs().sort_values("PC1", ascending=False).head(10).index
    rep.table(load.loc[top].reset_index(names="feature"), "pca_top_loadings_pc1")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(1, k + 1)
    ax.bar(x, 100 * p.explained_variance_ratio, color="#4C72B0", label="per component")
    ax.plot(
        x, 100 * np.cumsum(p.explained_variance_ratio), "o-", color="#C44E52", label="cumulative"
    )
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_xticks(x)
    ax.legend()
    ax.set_title(rep.title("Scree plot (PCA via numpy SVD, training patients)"))
    rep.figure(fig, "scree")

    S = p.transform(Z.to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for a in cfg["analysis"]["arms"]:
        m = (train.treatment == a).to_numpy()
        axes[0].scatter(S[m, 0], S[m, 1], s=9, alpha=0.6, color=COLORS[a], label=a)
    axes[0].legend(fontsize=8)
    axes[0].set_title("coloured by randomized arm")
    y = train[cfg["analysis"]["primary_outcome"]].to_numpy()
    obs = ~np.isnan(y)
    sc = axes[1].scatter(S[obs, 0], S[obs, 1], c=np.clip(y[obs], -20, 40), cmap="viridis", s=9)
    axes[1].scatter(
        S[~obs, 0],
        S[~obs, 1],
        s=9,
        facecolors="none",
        edgecolors="grey",
        label="week-52 vision missing",
    )
    fig.colorbar(sc, ax=axes[1], label="week-52 vision change (letters, clipped -20..40)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("coloured by outcome")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({100 * p.explained_variance_ratio[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * p.explained_variance_ratio[1]:.1f}%)")
    fig.suptitle(rep.title("Patient map on the first two principal components"))
    fig.tight_layout()
    rep.figure(fig, "pca_map")
    return {
        "pc1_explained": float(p.explained_variance_ratio[0]),
        "pc2_explained": float(p.explained_variance_ratio[1]),
        "pca_max_abs_diff_vs_sklearn": max(check.values()),
    }


def _cluster_section(Z: pd.DataFrame, train: pd.DataFrame, cfg, rep) -> dict[str, float]:
    res = cluster.fit_clusterings(Z.to_numpy(), cfg)
    rep.table(
        res["kmeans_scan"].merge(res["ward_scan"], on="k", suffixes=("_kmeans", "_ward")),
        "cluster_scan",
    )
    sizes = []
    effects = []
    for method in ("kmeans", "ward", "dbscan"):
        labels = res[method]
        sizes.append(
            pd.crosstab(labels, train.treatment).assign(method=method).reset_index(names="cluster")
        )
        effects.append(cluster.effects_by_cluster(train, labels, method, cfg))
    rep.table(pd.concat(sizes, ignore_index=True), "cluster_sizes")
    eff = pd.concat(effects, ignore_index=True)
    rep.table(eff, "cluster_effects")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    s = res["kmeans_scan"]
    axes[0].plot(s.k, s.inertia, "o-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("within-cluster sum of squares")
    axes[0].set_title("k-means elbow")
    axes[1].plot(s.k, s.silhouette, "o-", label="k-means")
    axes[1].plot(res["ward_scan"].k, res["ward_scan"].silhouette, "s--", label="Ward")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("mean silhouette")
    axes[1].legend()
    axes[1].set_title(f"silhouette (chosen k: k-means {res['k_kmeans']}, Ward {res['k_ward']})")
    fig.suptitle(rep.title("Choosing the number of clusters"))
    fig.tight_layout()
    rep.figure(fig, "cluster_choice")

    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(res["linkage"], truncate_mode="lastp", p=30, ax=ax, color_threshold=None)
    ax.set_xlabel("merged clusters (size in brackets)")
    ax.set_ylabel("Ward linkage distance")
    ax.set_title(rep.title("Hierarchical clustering dendrogram (Ward, last 30 merges)"))
    rep.figure(fig, "dendrogram")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(res["k_distance"])
    ax.axhline(res["eps"], color="#C44E52", ls="--", label=f"eps = {res['eps']:.2f}")
    ax.set_xlabel("training patients, sorted")
    ax.set_ylabel(f"distance to {cfg['represent']['dbscan_min_samples']}-th neighbour")
    ax.legend()
    ax.set_title(rep.title("DBSCAN k-distance plot"))
    rep.figure(fig, "dbscan_k_distance")

    e = eff.dropna(subset=["estimate"])
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(e) + 1.5))
    labels = [
        f"{r.method} c{r.cluster} (n={r.n_patients}): {r.comparison.split(' ')[0]}"
        for r in e.itertuples()
    ]
    ypos = np.arange(len(e))
    ax.errorbar(
        e.estimate,
        ypos,
        xerr=[e.estimate - e.ci_low, e.ci_high - e.estimate],
        fmt="o",
        capsize=3,
        color="#4C72B0",
    )
    ax.axvline(0, color="grey", lw=1)
    ax.axvline(cfg["analysis"]["mcid_letters"], color="grey", ls="--", lw=1, label="MCID")
    ax.set_yticks(ypos, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("difference vs bevacizumab in week-52 vision (letters, 95% CI)")
    ax.legend(fontsize=8)
    ax.set_title(rep.title("Treatment effect within clusters (EXPLORATORY)"))
    rep.figure(fig, "cluster_effects")
    return {
        "k_kmeans": res["k_kmeans"],
        "k_ward": res["k_ward"],
        "dbscan_n_clusters": float(len(set(res["dbscan"]) - {-1})),
        "dbscan_n_noise": float((res["dbscan"] == -1).sum()),
    }


def run(synthetic: bool) -> None:
    """Run M3 and log to MLflow experiment ``equipoise-m3-represent``."""
    sd = prepare(synthetic)
    cfg, train = sd.cfg, sd.train
    rep = tracking.Reporter(MODULE, synthetic, cfg)
    ct = build_preprocessor(cfg).fit(train)
    Z = transform_frame(ct, train)
    harm = ct.named_transformers_["cst"].named_steps["harmonize"]
    with_harm = train.assign(X_cst_um_harm=harm.transform(train[["X_cst_um", "X_oct_machine"]]))
    bands = discretize(with_harm, cfg)
    with tracking.start_run(EXPERIMENT, synthetic=synthetic, data_path=sd.data_path, cfg=cfg):
        tracking.log_params(
            {"represent": cfg["represent"], "n_train": len(train), "n_features": Z.shape[1]}
        )
        metrics = _pca_section(Z, train, cfg, rep)
        metrics |= _cluster_section(Z, train, cfg, rep)
        r = rules.rules_by_arm(train, bands, cfg)
        rep.table(r, "association_rules")
        metrics["n_rules"] = float(len(r))
        tracking.log_metrics(metrics)
    print(f"[represent] wrote {len(rep.written)} files")

"""Train/test split and cross-validation folds.

One fixed held-out test set (20 % by default) is drawn once, stratified by
randomized arm, with the seed from ``config/default.yaml``. It is saved to
``splits.json`` as patient IDs and is touched only for final reporting. All
model selection uses stratified K-fold cross-validation (or cross-fitting)
inside the remaining 80 %, so every fold keeps the trial's arm balance.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from equipoise.config import get_path, load_config, resolve_path
from equipoise.data.load import analysis_path

SPLITS_FILE = "splits.json"


@dataclass
class Splits:
    """The saved split: held-out test IDs, training IDs and CV fold assignment."""

    seed: int
    test_fraction: float
    n_folds: int
    stratify_by: str
    data_version: str
    train_ids: list[int]
    test_ids: list[int]
    #: fold index (0..n_folds-1) for each entry of ``train_ids``, same order
    train_fold: list[int]

    def fold_ids(self, k: int) -> tuple[list[int], list[int]]:
        """``(fit_ids, validation_ids)`` for fold ``k`` inside the training set."""
        ids, folds = np.asarray(self.train_ids), np.asarray(self.train_fold)
        return ids[folds != k].tolist(), ids[folds == k].tolist()


def make_test_split(
    df: pd.DataFrame,
    *,
    test_fraction: float | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Return sorted ``(train_ids, test_ids)`` patient IDs, stratified by arm."""
    cfg = cfg or load_config()
    test_fraction = cfg["split"]["test_fraction"] if test_fraction is None else test_fraction
    seed = cfg["seed"] if seed is None else seed
    stratify_by = stratify_by or cfg["split"]["stratify_by"]
    train_ids, test_ids = train_test_split(
        df["pt_id"].to_numpy(),
        test_size=test_fraction,
        random_state=seed,
        stratify=df[stratify_by].to_numpy(),
    )
    return sorted(int(i) for i in train_ids), sorted(int(i) for i in test_ids)


def cv_folds(
    df: pd.DataFrame,
    *,
    n_folds: int | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(fit_index, val_index)`` positional indices, stratified by arm.

    Use on the *training* rows only; the held-out test set never enters CV.
    """
    cfg = cfg or load_config()
    skf = StratifiedKFold(
        n_splits=n_folds or cfg["split"]["n_folds"],
        shuffle=True,
        random_state=cfg["seed"] if seed is None else seed,
    )
    yield from skf.split(np.zeros(len(df)), df[stratify_by or cfg["split"]["stratify_by"]])


def build_splits(df: pd.DataFrame, data_version: str, cfg: dict[str, Any] | None = None) -> Splits:
    """Draw the test split and assign CV folds to the training patients."""
    cfg = cfg or load_config()
    train_ids, test_ids = make_test_split(df, cfg=cfg)
    train = df.set_index("pt_id").loc[train_ids].reset_index()
    fold = np.empty(len(train), dtype=int)
    for k, (_, val) in enumerate(cv_folds(train, cfg=cfg)):
        fold[val] = k
    return Splits(
        seed=cfg["seed"],
        test_fraction=cfg["split"]["test_fraction"],
        n_folds=cfg["split"]["n_folds"],
        stratify_by=cfg["split"]["stratify_by"],
        data_version=data_version,
        train_ids=train_ids,
        test_ids=test_ids,
        train_fold=fold.tolist(),
    )


def splits_path(synthetic: bool, cfg: dict[str, Any] | None = None) -> Path:
    """``data/splits.json`` (real) or ``data/synthetic/splits.json``."""
    cfg = cfg or load_config()
    if synthetic:
        return resolve_path(cfg["paths"]["synthetic_dir"]) / SPLITS_FILE
    return get_path("splits_json", cfg)


def save_splits(splits: Splits, path: Path) -> Path:
    """Write the split as JSON (patient IDs only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(splits), indent=1) + "\n", encoding="utf-8")
    return path


def load_splits(
    synthetic: bool, df: pd.DataFrame | None = None, cfg: dict[str, Any] | None = None
) -> Splits:
    """Load the saved split; if ``df`` is given, check it covers exactly its patients."""
    path = splits_path(synthetic, cfg)
    if not path.exists():
        flag = " --synthetic" if synthetic else ""
        raise FileNotFoundError(f"{path} not found. Run `python -m equipoise split{flag}`.")
    s = Splits(**json.loads(path.read_text(encoding="utf-8")))
    if df is not None and set(s.train_ids) | set(s.test_ids) != set(df["pt_id"]):
        raise ValueError(f"{path} does not match the loaded table's patients; re-run the split.")
    return s


def split_frame(df: pd.DataFrame, splits: Splits) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)`` rows of ``df`` according to ``splits``."""
    is_test = df["pt_id"].isin(splits.test_ids)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def create_and_save(synthetic: bool, cfg: dict[str, Any] | None = None) -> Path:
    """CLI entry point: build the split for the chosen data source and save it."""
    from equipoise.data.load import load_analysis
    from equipoise.tracking import file_hash

    cfg = cfg or load_config()
    df = load_analysis(synthetic, cfg=cfg)
    splits = build_splits(df, file_hash(analysis_path(synthetic, cfg)), cfg)
    return save_splits(splits, splits_path(synthetic, cfg))

"""Shared set-up for pipeline stages: config, seeded data, the fixed split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equipoise.config import load_config, resolve_path
from equipoise.data.load import analysis_path, load_analysis
from equipoise.data.splits import Splits, create_and_save, load_splits, splits_path
from equipoise.utils import seed_everything


@dataclass
class StageData:
    """Everything a stage needs: config, the analysis table and the fixed split."""

    cfg: dict[str, Any]
    synthetic: bool
    df: pd.DataFrame
    splits: Splits
    data_path: Path

    @property
    def train_mask(self) -> np.ndarray:
        """Boolean mask of training (non-test) rows of ``df``."""
        return ~self.df["pt_id"].isin(self.splits.test_ids).to_numpy()

    @property
    def train(self) -> pd.DataFrame:
        """Training rows (the 80 %)."""
        return self.df[self.train_mask].reset_index(drop=True)

    def derived_dir(self) -> Path:
        """Folder for patient-level derived files (never committed)."""
        sub = "derived_synthetic" if self.synthetic else "derived"
        out = resolve_path("data") / sub
        out.mkdir(parents=True, exist_ok=True)
        return out


def prepare(synthetic: bool, cfg: dict[str, Any] | None = None) -> StageData:
    """Seed everything, load and validate data, load (or create) the fixed split."""
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    df = load_analysis(synthetic, cfg=cfg)
    if not splits_path(synthetic, cfg).exists():
        create_and_save(synthetic, cfg)
    splits = load_splits(synthetic, df, cfg)
    return StageData(cfg, synthetic, df, splits, analysis_path(synthetic, cfg))

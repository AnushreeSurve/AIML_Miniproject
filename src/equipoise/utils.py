"""Small shared utilities (reproducibility)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch; return a NumPy Generator.

    sklearn estimators and Optuna samplers take ``random_state`` / ``seed``
    explicitly from config; this function covers the global generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy global state, for libraries that still use it
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)

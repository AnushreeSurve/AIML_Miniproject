"""Single entry point for configuration.

All tunable numbers live in ``config/default.yaml``, ``config/costs.yaml`` and
``config/synthetic.yaml``. This module loads them into plain dictionaries,
resolves paths against the repository root, and decides which drug prices are
in force (real prices from ``costs.yaml``, an explicit override, or the
clearly labelled DEMO set).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

#: Repository root (``src/equipoise/config.py`` -> three levels up).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
#: Directory holding the YAML files; override with ``EQUIPOISE_CONFIG_DIR``.
CONFIG_DIR: Path = Path(os.environ.get("EQUIPOISE_CONFIG_DIR", REPO_ROOT / "config"))


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a dict (empty dict for an empty file)."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@cache
def _load_cached(name: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / f"{name}.yaml")


def load_config(name: str = "default") -> dict[str, Any]:
    """Return a deep copy of ``config/<name>.yaml`` so callers may mutate it."""
    return copy.deepcopy(_load_cached(name))


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path (relative to the repo root) to an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def get_path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    """Absolute path for ``paths.<key>`` in the default config."""
    cfg = cfg or load_config()
    return resolve_path(cfg["paths"][key])


# --------------------------------------------------------------------- prices
@dataclass(frozen=True)
class Prices:
    """Prices in force for the cost model.

    ``is_demo`` is True when the DEMO placeholder set is used; every output
    that shows a cost must then carry a visible warning.
    """

    per_injection: dict[str, float]
    laser_session: float
    is_demo: bool
    label: str
    units: str
    source: str | None = None
    warnings: list[str] = field(default_factory=list)


DEMO_WARNING = (
    "DEMO PRICES IN USE: costs are placeholder relative units, not real rupee prices. "
    "Fill config/costs.yaml (with a cited source) or pass --prices."
)


def resolve_prices(
    override: dict[str, Any] | None = None,
    costs_cfg: dict[str, Any] | None = None,
    arms: list[str] | None = None,
) -> Prices:
    """Decide which prices to use.

    Order of precedence: an explicit ``override`` (``{"per_injection": {...},
    "laser_session": x}``), then fully filled real prices in ``costs.yaml``,
    then the DEMO set. A partially filled ``costs.yaml`` is an error, because
    silently mixing real and placeholder prices would be misleading.
    """
    costs_cfg = costs_cfg if costs_cfg is not None else load_config("costs")
    arms = arms or load_config()["analysis"]["arms"]
    currency = costs_cfg.get("currency", "INR")

    if override is not None:
        per_inj = override.get("per_injection", {})
        missing = [a for a in arms if per_inj.get(a) is None]
        if missing or override.get("laser_session") is None:
            raise ValueError(
                f"--prices override must give per_injection for {arms} and laser_session; "
                f"missing: {missing or ['laser_session']}"
            )
        return Prices(
            per_injection={a: float(per_inj[a]) for a in arms},
            laser_session=float(override["laser_session"]),
            is_demo=False,
            label="OVERRIDE",
            units=currency,
            source="user override",
        )

    real = costs_cfg.get("prices", {})
    real_inj = real.get("price_per_injection_inr", {}) or {}
    values = [real_inj.get(a) for a in arms] + [real.get("laser_session_inr")]
    if all(v is not None for v in values):
        return Prices(
            per_injection={a: float(real_inj[a]) for a in arms},
            laser_session=float(real["laser_session_inr"]),
            is_demo=False,
            label="CONFIG",
            units=currency,
            source=real.get("source"),
        )
    if any(v is not None for v in values):
        raise ValueError(
            "config/costs.yaml is partially filled; fill every drug price and the laser "
            "price (with a source), or leave all of them null to use DEMO prices."
        )

    demo = costs_cfg["demo"]
    return Prices(
        per_injection={a: float(demo["price_per_injection"][a]) for a in arms},
        laser_session=float(demo["laser_session"]),
        is_demo=True,
        label=demo.get("label", "DEMO"),
        units=demo.get("units", "relative units"),
        source=None,
        warnings=[DEMO_WARNING],
    )

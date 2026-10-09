"""MLflow helpers shared by every stage.

Each module logs into its own experiment (``equipoise-m1-preprocess``,
``equipoise-m2-stats``, ...). Every run is tagged with the git commit, a short
hash of the input data file (the "data version") and whether synthetic data
was used. Figures and tables are written to ``reports/`` *and* attached to the
run, so the report and the MLflow UI always show the same artefacts.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: CI, Docker, Windows without a display
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from equipoise.config import REPO_ROOT, get_path, load_config, resolve_path  # noqa: E402

HASH_CHARS = 12
UNKNOWN = "unknown"


DB_NAME = "mlflow.db"
ARTIFACT_DIR = "artifacts"


def _store_dir(cfg: dict[str, Any]) -> Path | None:
    uri = str(cfg["tracking"]["mlflow_tracking_uri"])
    return None if ":" in uri and not Path(uri).drive else resolve_path(uri)


def tracking_uri(cfg: dict[str, Any] | None = None) -> str:
    """MLflow tracking URI from config.

    A plain directory (default ``mlruns``) becomes a SQLite store
    ``<dir>/mlflow.db``; MLflow 3 no longer accepts the bare file store. Any
    value containing a scheme (``sqlite:``, ``http://``) is used as-is.
    """
    cfg = cfg or load_config()
    store = _store_dir(cfg)
    if store is None:
        return str(cfg["tracking"]["mlflow_tracking_uri"])
    store.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(store / DB_NAME).as_posix()}"


def _ensure_experiment(name: str, cfg: dict[str, Any]) -> None:
    """Create the experiment with artifacts under ``<store>/artifacts/<name>`` if new."""
    if mlflow.get_experiment_by_name(name) is not None:
        return
    store = _store_dir(cfg)
    location = (store / ARTIFACT_DIR / name).as_uri() if store else None
    mlflow.create_experiment(name, artifact_location=location)


def experiment_name(module: str, cfg: dict[str, Any] | None = None) -> str:
    """``equipoise-<module>``, e.g. ``equipoise-m1-preprocess``."""
    cfg = cfg or load_config()
    return f"{cfg['tracking']['experiment_prefix']}-{module}"


def git_commit() -> str:
    """Return the git commit (``-dirty`` if the tree has changes), or ``unknown``."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN
    return f"{sha}-dirty" if dirty else sha


def file_hash(path: Path) -> str:
    """Short SHA-256 of a file, used as the data version tag."""
    if not path.exists():
        return UNKNOWN
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:HASH_CHARS]


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextmanager
def start_run(
    module: str,
    *,
    synthetic: bool,
    data_path: Path | None = None,
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run in the module's experiment with the standard tags."""
    cfg = cfg or load_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    name = experiment_name(module, cfg)
    _ensure_experiment(name, cfg)
    mlflow.set_experiment(name)
    std_tags = {
        "git_commit": git_commit(),
        "data_version": file_hash(data_path) if data_path else UNKNOWN,
        "data_source": "synthetic" if synthetic else "real",
        "module": module,
        "seed": str(cfg["seed"]),
    }
    with mlflow.start_run(run_name=run_name or module, tags={**std_tags, **(tags or {})}) as run:
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    """Log (nested) parameters; nested keys become ``a.b.c``."""
    mlflow.log_params({k: str(v) for k, v in _flatten(params).items()})


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log numeric metrics (non-numeric values are skipped)."""
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    mlflow.log_metrics(clean, step=step)


def report_dir(kind: str, synthetic: bool = False, cfg: dict[str, Any] | None = None) -> Path:
    """``reports/<kind>`` for real data, ``reports/synthetic/<kind>`` for synthetic runs."""
    cfg = cfg or load_config()
    if synthetic:
        return resolve_path(cfg["paths"]["synthetic_reports_dir"]) / kind
    return get_path(f"{kind}_dir", cfg)


def save_figure(
    fig: Figure,
    module: str,
    name: str,
    *,
    log: bool = True,
    synthetic: bool = False,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Save ``reports/figures/<module>_<name>.png`` and attach it to the active run."""
    out = report_dir("figures", synthetic, cfg) / f"{module}_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if log and mlflow.active_run():
        mlflow.log_artifact(str(out), artifact_path="figures")
    return out


def save_table(
    df: pd.DataFrame,
    module: str,
    name: str,
    *,
    log: bool = True,
    index: bool = False,
    synthetic: bool = False,
    cfg: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save ``reports/tables/<module>_<name>.csv`` and ``.md`` and attach both to the run."""
    base = report_dir("tables", synthetic, cfg) / f"{module}_{name}"
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = base.with_suffix(".csv"), base.with_suffix(".md")
    df.to_csv(csv_path, index=index)
    md_path.write_text(df.to_markdown(index=index) + "\n", encoding="utf-8")
    if log and mlflow.active_run():
        mlflow.log_artifact(str(csv_path), artifact_path="tables")
        mlflow.log_artifact(str(md_path), artifact_path="tables")
    return csv_path, md_path


@dataclass
class Reporter:
    """Per-stage writer for figures and tables (real vs synthetic output folders).

    Figures from synthetic runs get a "[SYNTHETIC DATA]" title suffix so they
    can never be mistaken for trial results.
    """

    module: str
    synthetic: bool
    cfg: dict[str, Any]
    written: list[Path] = field(default_factory=list)

    def title(self, text: str) -> str:
        """Figure title, marked when the data are synthetic."""
        return f"{text} [SYNTHETIC DATA]" if self.synthetic else text

    def table(self, df: pd.DataFrame, name: str, *, index: bool = False) -> Path:
        """Write ``<module>_<name>.csv/.md`` and log both."""
        csv, md = save_table(
            df, self.module, name, index=index, synthetic=self.synthetic, cfg=self.cfg
        )
        self.written += [csv, md]
        return csv

    def figure(self, fig: Figure, name: str) -> Path:
        """Write ``<module>_<name>.png``, log it and close the figure."""
        import matplotlib.pyplot as plt

        out = save_figure(fig, self.module, name, synthetic=self.synthetic, cfg=self.cfg)
        plt.close(fig)
        self.written.append(out)
        return out


def child_run(name: str, params: Mapping[str, Any], metrics: Mapping[str, float]) -> None:
    """Log one model as a nested MLflow run under the active stage run."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tag("model", name)
        log_params(params)
        log_metrics(metrics)

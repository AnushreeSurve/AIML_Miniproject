"""Command-line interface: ``python -m equipoise <stage> [--synthetic]``.

Stages run in README section 6.4 order. ``all`` runs every pipeline stage in
sequence. Stages that are not built yet print which phase adds them and exit
successfully, so ``all`` is always runnable.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import typer

from equipoise.config import load_config
from equipoise.utils import seed_everything

app = typer.Typer(add_completion=False, help="Equipoise pipeline (research prototype).")

SyntheticOpt = typer.Option(False, "--synthetic", help="Use data/synthetic/ instead of real data.")


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: CLI name, MLflow module id and the phase that implements it."""

    name: str
    module: str
    phase: int
    help: str
    run: str | None = None  # "package.module:function", imported lazily


STAGES: tuple[Stage, ...] = (
    Stage(
        "preprocess",
        "m1-preprocess",
        2,
        "M1: missing data, harmonization, outliers, encoding",
        "equipoise.preprocess.stage:run",
    ),
    Stage(
        "stats",
        "m2-stats",
        2,
        "M2: Table 1, trial replication, hypothesis tests",
        "equipoise.stats.stage:run",
    ),
    Stage("represent", "m3-represent", 3, "M3: PCA/SVD, clustering, association rules"),
    Stage("supervised", "m5-supervised", 3, "M5: regression, classification, count models"),
    Stage("search", "m4-search", 4, "M4: feature selection and search arena"),
    Stage("causal", "m6-causal", 5, "M6: counterfactual engine and policy value"),
    Stage("longitudinal", "m7-longitudinal", 6, "M7: trajectories, time series, HMM"),
    Stage("decision", "m8-decision", 7, "M8: cost model, break-even, fuzzy, expert system"),
    Stage("explain", "m9-explain", 7, "M9: SHAP and subgroup checks"),
    Stage("report-assets", "m11-report", 11, "Report tables, flow diagram, asset map"),
)
#: Not part of ``all``: monitoring runs against logged API requests.
MONITOR = Stage("monitor", "m11-monitor", 9, "Evidently drift report on logged API requests")


def _run_stage(stage: Stage, synthetic: bool) -> None:
    seed_everything(load_config()["seed"])
    if stage.run is None:
        typer.echo(f"[{stage.name}] not implemented yet (Phase {stage.phase}).")
        return
    module, func = stage.run.split(":")
    getattr(importlib.import_module(module), func)(synthetic)


def _register(stage: Stage) -> None:
    def command(synthetic: bool = SyntheticOpt) -> None:
        _run_stage(stage, synthetic)

    command.__doc__ = stage.help
    app.command(name=stage.name, help=stage.help)(command)


for _stage in (*STAGES, MONITOR):
    _register(_stage)


@app.command(name="all")
def run_all(synthetic: bool = SyntheticOpt) -> None:
    """Run every pipeline stage in order."""
    for stage in STAGES:
        _run_stage(stage, synthetic)


@app.command()
def split(synthetic: bool = SyntheticOpt) -> None:
    """Draw the fixed train/test split and CV folds and save them to splits.json."""
    from equipoise.data.splits import create_and_save, load_splits

    seed_everything(load_config()["seed"])
    path = create_and_save(synthetic)
    s = load_splits(synthetic)
    typer.echo(
        f"wrote {path}: {len(s.train_ids)} train / {len(s.test_ids)} test, {s.n_folds} folds"
    )


@app.command()
def validate(synthetic: bool = SyntheticOpt) -> None:
    """Validate the analysis (and visit, if present) table against the schema."""
    from equipoise.data.load import load_analysis, load_visits, visits_path

    df = load_analysis(synthetic)
    typer.echo(f"analysis table OK: {len(df)} patients, {df['treatment'].value_counts().to_dict()}")
    if visits_path(synthetic).exists():
        v = load_visits(synthetic, analysis=df)
        typer.echo(f"visit table OK: {len(v)} rows")
    else:
        typer.echo("visit table not built yet (Phase 1).")


if __name__ == "__main__":
    app()

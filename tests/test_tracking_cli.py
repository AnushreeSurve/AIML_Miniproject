import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from typer.testing import CliRunner

from equipoise import tracking
from equipoise.cli import STAGES, app
from equipoise.data.load import analysis_path


def test_run_logs_tags_tables_and_figures(tmp_path, cfg):
    cfg = {
        **cfg,
        "tracking": {**cfg["tracking"], "mlflow_tracking_uri": str(tmp_path / "mlruns")},
        "paths": {
            **cfg["paths"],
            "figures_dir": str(tmp_path / "fig"),
            "tables_dir": str(tmp_path / "tab"),
        },
    }
    with tracking.start_run(
        "m0-test", synthetic=True, data_path=analysis_path(True), cfg=cfg
    ) as run:
        tracking.log_params({"a": {"b": 1}})
        tracking.log_metrics({"rmse": 1.5, "note": "skip"})
        csv, md = tracking.save_table(pd.DataFrame({"x": [1]}), "m0", "demo", cfg=cfg)
        fig, ax = plt.subplots()
        png = tracking.save_figure(fig, "m0", "demo", cfg=cfg)
    assert csv.exists() and md.exists() and png.exists()
    r = mlflow.get_run(run.info.run_id)
    assert r.data.params["a.b"] == "1" and r.data.metrics["rmse"] == 1.5
    assert r.data.tags["data_source"] == "synthetic"
    assert len(r.data.tags["data_version"]) == tracking.HASH_CHARS
    exp = mlflow.get_experiment(r.info.experiment_id)
    assert exp.name == "equipoise-m0-test"


def test_cli_lists_all_stages():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    for s in [*(s.name for s in STAGES), "monitor", "all", "validate"]:
        assert s in res.output


def test_cli_all_and_validate_on_synthetic(fast_config):
    runner = CliRunner()
    assert runner.invoke(app, ["all", "--synthetic"]).exit_code == 0
    res = runner.invoke(app, ["validate", "--synthetic"])
    assert res.exit_code == 0, res.output
    assert "analysis table OK" in res.output


def test_outputs_written_only_where_configured(tmp_path, cfg):
    cfg = {**cfg, "paths": {**cfg["paths"], "tables_dir": str(tmp_path / "t")}}
    csv, _ = tracking.save_table(pd.DataFrame({"x": [1]}), "m0", "probe", log=False, cfg=cfg)
    assert csv.parent == tmp_path / "t"

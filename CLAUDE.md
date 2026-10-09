# CLAUDE.md: working rules for AI coding assistants on Equipoise

`README.md` is the specification. Read it before changing anything. This is a
college AIML mini-project and a **research prototype, not for clinical use**.

## Process

- Work **one phase at a time** (phases 0–11 are in the team's build brief). At the end of each phase:
  1. run `pytest` and the phase's commands,
  2. fix failures,
  3. commit (`feat(mN): ...`),
  4. summarise: what was built, the commands, the outputs to look at, and any open doubts.
  5. Then **stop and wait** for "continue".
- **Raw DRCR data: ask, don't guess.** Inspect the actual files and `scripts/build_protocolT_table.py` for column names. If something is ambiguous, check the CRFs / Protocol PDF, otherwise ask.
- **Never push to GitHub** unless the team explicitly allows it. Never delete files outside the repo.
- **Never commit anything under `data/`** except `data/synthetic/` and DVC pointer files (`data/*.dvc`, which hold only a hash and size). Raw and derived DRCR data must not be redistributed.

## Scientific integrity (non-negotiable)

- **Model inputs are `X_` columns only.**
  - `Y_` columns are targets; `M_` columns are bookkeeping.
  - `treatment` is allowed only where a learner explicitly takes it as the treatment variable.
  - Post-randomization raw tables are **never** features: `TAdvEvent`, `TMedication`, `TMedUpFU`, `TFUExamQuestions`, `TOcularExamFU`, `TVisitComp`, `TNonStudyEyeInj`, `TStudyProcedure`, `TPostRandPtFinalStatus`.
  - The single exception is M7's early-response model, which may use vision up to week 12. It lives in its own module and is named clearly.
- **Every function that builds a baseline feature matrix** must be registered with `@register_feature_builder` (`equipoise.data.features`). `tests/test_leakage.py` then checks it automatically.
- **Fit preprocessing on training folds only.** This covers imputers, scalers, encoders and harmonization. Use sklearn `Pipeline` / `ColumnTransformer`.
- **Splits.**
  - One fixed held-out test set: 20%, stratified by `treatment`, seed from config (`equipoise.data.splits`).
  - Touch it only for final reporting.
  - Use CV / cross-fitting inside the 80%. **Never tune on the test set.**
- **Never fabricate or tidy results.**
  - "No heterogeneous effect" or "no better than baseline" are legitimate findings; report them plainly.
  - Only numbers produced by the code go into docs or the README.
- **Seeds everywhere** (numpy, random, torch, sklearn, optuna): `equipoise.utils.seed_everything` plus `random_state` from config. Results must reproduce run to run.

## Domain facts

| Item | Value |
|---|---|
| Arms | Aflibercept 224, Bevacizumab 218, Ranibizumab 218 (total 660) |
| Primary outcome | `Y_va_change_wk52` (ETDRS letters) |
| Missing week-52 vision | 44 patients |
| Reference arm | **Bevacizumab** (cheapest); report aflibercept − bevacizumab and ranibizumab − bevacizumab |
| Equipoise threshold | MCID = 5 letters (`analysis.mcid_letters`) |
| Propensities | known from randomization: use the empirical arm proportions, **never** a fitted model |
| Published subgroup | `X_va_below_69 = 1` ⇔ `X_va_letters <= 68` (20/50 or worse) |
| Replication targets | README §9.1 (complete-case means within 0.2 letters) |

## Prices

- **Never invent prices.** `config/costs.yaml` keeps every price `null` until the team fills it in with a cited source.
- With `null` prices, `equipoise.config.resolve_prices()` uses an explicit `--prices` override or the labelled **DEMO** set (relative units).
- Every output and app page must show a visible warning whenever demo prices are in use (`Prices.is_demo`).

## Code conventions

- **Package and CLI.** Code lives in `src/equipoise/`. The CLI is `python -m equipoise <stage> [--synthetic]` (`cli.py`, typer). Notebooks stay thin.
- **Config.** All tunables are in `config/default.yaml`, `config/costs.yaml` and `config/synthetic.yaml`, loaded only through `equipoise.config`. No magic numbers in code.
- **Cross-platform.** `pathlib` everywhere, no shell-only tricks, no `make`. The team may be on Windows. Python 3.11.
- **Style.** Type hints and docstrings; `ruff check .` and `ruff format --check .` must pass.
- **Module docstrings** explain the method in 2–4 plain sentences, defensible in a viva.
- **Stage outputs.** Every stage writes:
  - figures to `reports/figures/<module>_*.png`,
  - tables to `reports/tables/<module>_*.csv` **and** `.md` (use `tracking.save_figure` / `tracking.save_table`),
  - one MLflow run per module experiment `equipoise-mN-<stage>` (`tracking.start_run`).
- **MLflow store.** `mlruns/mlflow.db` (SQLite; MLflow 3 rejects the bare file store) with artifacts under `mlruns/artifacts/`. View it with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Dependencies.** Keep core lean; heavy pieces go in extras `[automl]` / `[cv]`. Pins are in `requirements*.txt`, generated with `uv pip compile pyproject.toml --universal --python-version 3.11` (add `--extra dev` for `requirements-dev.txt`).
- **Synthetic data.** `python scripts/make_synthetic_data.py` regenerates `data/synthetic/` deterministically from `config/synthetic.yaml`. It has a planted effect: aflibercept helps more when `X_va_below_69 = 1`, and the truth is in `protocolT_truth.csv`. Tests and CI use only synthetic data. Never report synthetic numbers as results.

## Data pipeline (DVC)

- `dvc.yaml` holds the stages; `dvc repro` runs them. The cache is local, with no remote.
- `data/protocolT_analysis.csv` is tracked by `dvc add` until the M0 build script and `data/raw/` are in the repo.
- Then switch to the commented `build_analysis` / `build_visits` stages in `dvc.yaml`.
- `data/splits.json` (real) and `data/synthetic/splits.json` hold the fixed test split and CV folds (`python -m equipoise split [--synthetic]`). Load them with `equipoise.data.splits.load_splits`. Never redraw splits ad hoc.
- `scripts/build_protocolT_visits.py` reads raw column names only from `config/raw_tables.yaml`, which must be filled by inspecting the files. Study eye comes from the M0 table's `study_eye`.

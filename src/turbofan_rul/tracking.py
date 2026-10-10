"""Experiment tracking with MLflow: one run per model scored under the development protocol.

Runs are stored locally (SQLite database and artifact folder at the project root, both
gitignored). Browse them with:  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
import subprocess
import tempfile
from pathlib import Path

# Silence the hint MLflow prints on import; it is not relevant to this project.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

from turbofan_rul.config import (
    MIN_HISTORY,
    N_CUTS_PER_ENGINE,
    N_FOLDS,
    PLANNING_LIMIT,
    PROJECT_ROOT,
    RANDOM_SEED,
    RUL_CAP,
)

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()


def start_experiment(
    name: str, tracking_uri: str = TRACKING_URI, artifact_root: str = ARTIFACT_ROOT
) -> str:
    """Point MLflow at the project store and select (or create) an experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_root)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def protocol_params() -> dict:
    """The protocol settings every run is scored under, so runs can be filtered by them."""
    return {
        "subset": "FD001",
        "n_folds": N_FOLDS,
        "min_history": MIN_HISTORY,
        "planning_limit": PLANNING_LIMIT,
        "n_cuts_per_engine": N_CUTS_PER_ENGINE,
        "seed": RANDOM_SEED,
        "rul_cap": RUL_CAP,
    }


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def delete_runs_named(run_name: str) -> int:
    """Delete every run with this name in the current experiment. Returns how many."""
    experiment = mlflow.get_experiment_by_name(mlflow.get_experiment(_current_experiment_id()).name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="list",
    )
    for run in runs:
        mlflow.delete_run(run.info.run_id)
    return len(runs)


def _current_experiment_id() -> str:
    return mlflow.tracking.fluent._get_experiment_id()


def log_model_run(
    run_name: str,
    params: dict,
    metrics: dict,
    predictions: pd.DataFrame | None = None,
    tags: dict | None = None,
    replace: bool = True,
) -> str:
    """Log one scored model: protocol and model parameters, metrics, and OOF predictions.

    With `replace`, earlier runs with the same name are deleted first, so re-running a
    notebook updates its runs instead of adding duplicates.
    """
    if replace:
        delete_runs_named(run_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**protocol_params(), **params})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"git_commit": git_commit(), **(tags or {})})
        if predictions is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "oof_predictions.csv"
                predictions.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
        return run.info.run_id

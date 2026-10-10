import mlflow
import pandas as pd

from turbofan_rul.tracking import log_model_run, start_experiment


def test_run_is_logged_with_protocol_params_and_predictions(tmp_path):
    start_experiment(
        "test", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=(tmp_path / "art").as_uri()
    )
    preds = pd.DataFrame({"unit": [1, 2], "cycle": [50, 60], "rul_true": [10, 20], "pred": [12, 18]})
    run_id = log_model_run("toy", {"model": "toy"}, {"rmse_mean": 2.0}, predictions=preds)

    run = mlflow.get_run(run_id)
    assert run.data.params["model"] == "toy"
    assert run.data.params["n_folds"] == "5"
    assert run.data.metrics["rmse_mean"] == 2.0
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)] == ["oof_predictions.csv"]


def test_relogging_replaces_the_previous_run(tmp_path):
    start_experiment(
        "test2", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=(tmp_path / "art").as_uri()
    )
    log_model_run("same-name", {"model": "v1"}, {"rmse_mean": 3.0})
    log_model_run("same-name", {"model": "v2"}, {"rmse_mean": 2.0})
    runs = mlflow.search_runs(experiment_names=["test2"])
    assert len(runs) == 1 and runs.loc[0, "params.model"] == "v2"

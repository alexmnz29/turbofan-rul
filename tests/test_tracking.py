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

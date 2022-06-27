import os
import json
import pickle
from pathlib import Path

import click
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@click.command("valid")
@click.option("--data-dir")
@click.option("--metrics-dir")
def main(data_dir: str, metrics_dir: str):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL", "mlflow:5001"))
    mlflow.set_experiment(f"LogisticRegression_valid")
    # mlflow.sklearn.autolog()  # enable autologging
    client = MlflowClient()

    data_dir = Path(data_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    last_model_params = sorted(
        client.search_model_versions("name='LogisticRegression'"),
        key=lambda x: x.last_updated_timestamp,
    )[-1]
    model = mlflow.sklearn.load_model(f"models:/{last_model_params.name}/{last_model_params.version}")

    # Validate
    valid_data = pd.read_csv(data_dir / "valid.csv")
    valid_features, valid_targets = valid_data.drop("target", axis=1), valid_data.target

    preds = model.predict(valid_features)
    metrics = {
        "accuracy": accuracy_score(valid_targets, preds),
        "f1_score": f1_score(valid_targets, preds),
        "AUC": roc_auc_score(valid_targets, preds),
    }
    with open(metrics_dir / "metrics.json", "w") as fout:
        json.dump(metrics, fout)

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        
    client.transition_model_version_stage(
        name=last_model_params.name,
        version=last_model_params.version,
        stage="Production"
    )


if __name__ == '__main__':
    main()
 
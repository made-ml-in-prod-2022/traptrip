import os
import pickle
from pathlib import Path

import click
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--save-model-dir")
def main(data_dir: str, save_model_dir: str):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL", "mlflow:5001"))
    mlflow.set_experiment(f"LogisticRegression_train")
    mlflow.sklearn.autolog()  # enable autologging
    with mlflow.start_run():
        data_dir = Path(data_dir)
        save_model_dir = Path(save_model_dir)
        save_model_dir.mkdir(parents=True, exist_ok=True)

        params = {"C": 10, "solver": "sag"}
        model = LogisticRegression(**params)

        train_data = pd.read_csv(data_dir / "train.csv")
        train_features, train_targets = train_data.drop("target", axis=1), train_data.target

        model.fit(train_features, train_targets)
        with open(save_model_dir / "model.pkl", "wb") as fout:
            pickle.dump(model, fout)

        mlflow.sklearn.log_model(model, artifact_path="models", registered_model_name="LogisticRegression")

if __name__ == '__main__':
    main()
 
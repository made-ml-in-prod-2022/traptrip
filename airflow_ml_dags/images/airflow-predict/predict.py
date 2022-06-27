import os
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import click
import mlflow


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--data-transformers-dir")
def main(input_dir: str, output_dir: str, data_transformers_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    data_transformers_dir = Path(data_transformers_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_transformer_dir = sorted(
        list(Path(data_transformers_dir).iterdir()), 
        key=lambda p: datetime.strptime(p.name, "%Y-%m-%d")
    ).pop()

    data = pd.read_csv(str(input_dir / "data.csv"))
    with open(data_transformer_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL", "mlflow:5001"))
    model = mlflow.pyfunc.load_model(model_uri=f"models:/LogisticRegression/Production")
    data["target"] = model.predict(scaler.transform(data))

    data.to_csv(str(output_dir / "predictions.csv"), index=False)


if __name__ == "__main__":
    main()

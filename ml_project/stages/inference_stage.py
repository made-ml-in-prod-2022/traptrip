import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml_project.entities import Config
from ml_project.utils.ml_utils import load_pickle
from ml_project.preprocessing import Dataset
from ml_project.utils.technical_utils import get_last_artifacts_path


def get_checkpoint_path(cfg: Config) -> Path:
    checkpoint_path = (
        Path(cfg.general.project_dir)
        / cfg.general.artifacts_dir
        / cfg.inference.run_name
        / cfg.general.checkpoint_path
    )
    if not checkpoint_path.exists():
        logging.debug(
            f"There is no artifacts in {checkpoint_path}. Trying to get last artifacts path"
        )
        last_artifacts_dir = get_last_artifacts_path(cfg)
        if last_artifacts_dir:
            checkpoint_path = last_artifacts_dir / cfg.general.checkpoint_path
            logging.debug(f"Using last experiment with artifacts {checkpoint_path}")
        else:
            raise FileNotFoundError("There is no artifacts in stated path")
    return checkpoint_path


def load_model(checkpoint_path: Path) -> Any:
    model = load_pickle(checkpoint_path / "model.pkl")
    return model


def get_data(cfg: Config) -> pd.DataFrame:
    data, _ = Dataset(cfg).load_dataset()
    return data


def make_prediction(model: Any, data: pd.DataFrame, save_path: str):
    prediction = model.predict(data)
    with open(save_path, "w") as fout:
        fout.write("\n".join(prediction.astype(str)))

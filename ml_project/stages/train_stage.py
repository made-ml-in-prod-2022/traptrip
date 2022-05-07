import os
import logging
import pickle
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

from ml_project.entities.config import Config
from ml_project.preprocessing import Dataset
from ml_project.utils.technical_utils import load_obj


def get_data(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    data, target = Dataset(cfg.dataset).load_dataset()
    data_transformer = instantiate(cfg.preprocessing)
    data = data_transformer.fit_transform(data, target)
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=cfg.dataset.test_size, random_state=cfg.general.seed
    )
    return train_data, test_data, train_target, test_target, data_transformer


def train_model(cfg: Config, data: pd.DataFrame, target: pd.Series) -> Any:
    model = instantiate(cfg.model)
    model.fit(data, target)
    return model


def get_score(cfg: Config, model: Any, data: pd.DataFrame, target: pd.Series) -> float:
    metric = load_obj(cfg.metric._target_)
    score = metric(target, model.predict(data))
    logging.info(f"{metric.__name__}: {score:.6f}")
    return score


def save_artifacts(
    cfg: Config, model: Any, data_transformer: Any, score: float
) -> None:
    Path(cfg.general.checkpoint_path).mkdir(exist_ok=True)
    model_path = os.path.join(cfg.general.checkpoint_path, "model.pkl")
    transformer_path = os.path.join(cfg.general.checkpoint_path, "data_transformer.pkl")
    score_path = os.path.join(cfg.general.checkpoint_path, "score.txt")
    with open(model_path, "wb") as fout:
        pickle.dump(model, fout)
    with open(transformer_path, "wb") as fout:
        pickle.dump(data_transformer, fout)
    with open(score_path, "w") as fout:
        fout.write(f"{load_obj(cfg.metric._target_).__name__}: {score}")
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Data transformer saved to {transformer_path}")
    logging.info(f"Score saved to {score_path}")

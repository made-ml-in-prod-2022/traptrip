import os
import logging
import pickle
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_project.entities.config import Config
from ml_project.preprocessing import Dataset
from ml_project.utils.technical_utils import load_obj


def get_data(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    data, target = Dataset(cfg.dataset).load_dataset()
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=cfg.dataset.test_size, random_state=cfg.general.seed
    )
    return train_data, test_data, train_target, test_target


def train_model(cfg: Config, data: pd.DataFrame, target: pd.Series) -> Any:
    model = instantiate(cfg.model)
    data_transformer = instantiate(cfg.preprocessing)
    model_pipeline = Pipeline(
        steps=[("data_transformer", data_transformer), ("classifier", model)]
    )
    model_pipeline.fit(data, target)
    return model_pipeline


def get_score(cfg: Config, model: Any, data: pd.DataFrame, target: pd.Series) -> float:
    metric = load_obj(cfg.metric._target_)
    score = metric(target, model.predict(data))
    logging.info(f"{metric.__name__}: {score:.6f}")
    return score


def save_artifacts(cfg: Config, model: Any, score: float, logger: Any) -> None:
    Path(cfg.general.checkpoint_path).mkdir(exist_ok=True)
    model_path = os.path.join(cfg.general.checkpoint_path, "model.pkl")
    score_path = os.path.join(cfg.general.checkpoint_path, "score.txt")
    with open(model_path, "wb") as fout:
        pickle.dump(model, fout)
    with open(score_path, "w") as fout:
        fout.write(f"{load_obj(cfg.metric._target_).__name__}: {score}")
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Score saved to {score_path}")

    if logger:
        logger.log_metric(load_obj(cfg.metric._target_).__name__, score)
        logger.log_sklearn_model(model, cfg.general.checkpoint_path)
        logger.log_params(cfg.model)
        logger.end_run()


def initialize_logger(cfg: Config):
    logger = instantiate(cfg.logger) if "logger" in cfg else None
    return logger

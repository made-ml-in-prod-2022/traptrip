import os
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

from utils.utils import load_obj


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")


# @dataclass
# class LogregConfig:
#     _target_: str = "sklearn.linear_model.LogisticRegression"
#     penalty: str = "l1"
#     solver: str = "liblinear"
#     C: float = 1.0
#     random_state: int = 42
#     max_iter: int = 1000


# @dataclass
# class AccuracyConfig:
#     _target_: str = "sklearn.metrics.accuracy_score"


# @dataclass
# class F1scoreConfig:
#     _target_: str = "sklearn.metrics.f1_score"
#     average: str = "binary"


# @dataclass
# class HeartDataConfig:
#     test_size: float = 0.2
#     random_state: int = 42


# @dataclass
# class GeneralConfig:
#     seed: int = 42


# @dataclass
# class Config:
#     # We will populate db using composition.
#     model: Any = LogregConfig()
#     metric: Any = F1scoreConfig()
#     general: GeneralConfig = GeneralConfig()


# cs = ConfigStore.instance()
# cs.store(name="config", node=Config)
# cs.store(group="model", name="logreg", node=LogregConfig)
# cs.store(group="metric", name="accuracy", node=AccuracyConfig)
# cs.store(group="metric", name="f1_score", node=F1scoreConfig)


def download_dataset(save_path: str, download_path: str, **kwargs) -> None:
    os.system(
        f"kaggle datasets download {download_path} && "
        f"unzip {download_path.split('/')[-1]} -d {os.path.dirname(save_path)} -y && "
        f"rm {download_path.split('/')[-1]}.zip"
    )


def load_dataset(
    datapath: str, target_column: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    datapath = os.path.join(get_original_cwd(), datapath)
    if not os.path.exists(datapath):
        download_dataset(datapath, **kwargs)
    df = pd.read_csv(datapath)
    dataset, target = df.drop(target_column, axis=1), df[target_column]
    return dataset, target


@hydra.main(config_path="conf", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset, target = load_dataset(**cfg.dataset)
    train_data, test_data, train_target, test_target = train_test_split(
        dataset, target, test_size=cfg.dataset.test_size, random_state=cfg.general.seed
    )

    model = instantiate(cfg.model)
    model.fit(train_data, train_target)

    metric = load_obj(cfg.metric._target_)
    score = metric(test_target, model.predict(test_data))
    logging.info(f"{metric.__name__}: {score:.6f}")

    with open(cfg.general.checkpoint_name, "wb") as fout:
        pickle.dump(model, fout)
    logging.info(f"Model saved to {cfg.general.checkpoint_name}")


if __name__ == "__main__":
    run()

from sklearn.linear_model import LogisticRegression

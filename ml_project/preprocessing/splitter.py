from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


def split_data(
    data: pd.DataFrame, target: pd.Series, cfg: DictConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=cfg.dataset.test_size, random_state=cfg.general.seed
    )
    return train_data, test_data, train_target, test_target

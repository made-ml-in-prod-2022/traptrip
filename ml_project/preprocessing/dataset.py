import os
from typing import Tuple

import pandas as pd
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from preprocessing.data_transformer import DefaultTransformer


class Dataset:
    def __init__(self, cfg: DictConfig) -> None:
        self.download_path = cfg.download_path
        self.datapath = cfg.datapath
        self.target_column = cfg.target_column

    def download_dataset(self) -> None:
        os.system(
            f"kaggle datasets download {self.download_path} && "
            f"unzip {self.download_path.split('/')[-1]} -d {os.path.dirname(self.datapath)} -y && "
            f"rm {self.download_path.split('/')[-1]}.zip"
        )

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        if not os.path.exists(self.datapath):
            self.download_dataset()
        df = pd.read_csv(self.datapath)
        data, target = df.drop(self.target_column, axis=1), df[self.target_column]
        return data, target

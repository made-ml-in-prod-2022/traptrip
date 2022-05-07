import os
import logging
from typing import Tuple

import pandas as pd

from ml_project.entities import Config


class Dataset:
    def __init__(self, cfg: Config) -> None:
        self.download_path = cfg.download_path
        self.data_dir = cfg.data_dir
        self.datapath = cfg.datapath
        self.target_column = cfg.target_column

    def download_dataset(self) -> None:
        os.system(
            f"kaggle datasets download {self.download_path} && "
            f"unzip {self.download_path.split('/')[-1]} -d {self.data_dir} && "
            f"rm {self.download_path.split('/')[-1]}.zip"
        )

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        if not os.path.exists(self.datapath):
            logging.info(f"Dataset isn't downloaded. Downloading in {self.datapath}")
            self.download_dataset()
        df = pd.read_csv(self.datapath)
        data, target = df.drop(self.target_column, axis=1), df[self.target_column]
        return data, target

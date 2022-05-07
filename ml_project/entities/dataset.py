import os
from dataclasses import dataclass
from typing import Optional, List, Any


@dataclass
class DatasetConfig:
    test_size: float = 0.2
    random_state: int = 42
    download_path: str = "cherngs/heart-disease-cleveland-uci"
    data_dir: str = os.path.join(os.getcwd(), "data")
    datapath: str = os.path.join(os.getcwd(), "data", "heart_cleveland_upload.csv")
    target_column: str = "condition"


@dataclass
class PreprocessingConfig:
    _target_: str = "preprocessing.data_transformer.DefaultTransformer"
    numerical: Optional[List[Any]] = None
    categorial: Optional[List[Any]] = None

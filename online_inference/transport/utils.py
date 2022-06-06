from pathlib import Path
from dataclasses import dataclass
from typing import Union

import yaml
import pickle


@dataclass
class Config:
    host: str
    port: int
    model_path: str
    log_level: str = "INFO"


def load_cfg(cfg_path: Path):
    return Config(**yaml.safe_load(cfg_path.read_text("utf-8")))


def load_pkl(obj_path: Union[str, Path]):
    with open(obj_path, "rb") as f:
        return pickle.load(f)

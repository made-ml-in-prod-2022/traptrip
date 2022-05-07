import os
from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from ml_project.entities import (
    LogregConfig,
    F1scoreConfig,
    AccuracyConfig,
    DatasetConfig,
    PreprocessingConfig,
    RfConfig,
)


@dataclass
class GeneralConfig:
    seed: int = 42
    project_dir: str = os.getcwd()
    checkpoint_path: str = "weights"


@dataclass
class Config:
    model: Any = LogregConfig()
    metric: Any = F1scoreConfig()
    dataset: Any = DatasetConfig()
    preprocessing: Any = PreprocessingConfig()
    general: Any = GeneralConfig()


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="conf/config", node=Config)
    cs.store(group="conf/model", name="logreg", node=LogregConfig)
    cs.store(group="conf/model", name="rf", node=RfConfig)
    cs.store(group="conf/metric", name="accuracy", node=AccuracyConfig)
    cs.store(group="conf/metric", name="f1_score", node=F1scoreConfig)
    cs.store(group="conf/dataset", name="heart", node=DatasetConfig)
    cs.store(group="conf/preprocessing", name="default", node=PreprocessingConfig)
    cs.store(name="conf/general", node=GeneralConfig)

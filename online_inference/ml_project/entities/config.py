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
    DefaultInferenceConfig,
    MLflowLoggerConfig,
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
    general: GeneralConfig = GeneralConfig()
    inference: Any = DefaultInferenceConfig()
    logger: Any = None


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="conf/config", node=Config)
    cs.store(name="conf/general", node=GeneralConfig)
    cs.store(group="conf/model", name="logreg", node=LogregConfig)
    cs.store(group="conf/model", name="rf", node=RfConfig)
    cs.store(group="conf/metric", name="accuracy", node=AccuracyConfig)
    cs.store(group="conf/metric", name="f1_score", node=F1scoreConfig)
    cs.store(group="conf/dataset", name="heart", node=DatasetConfig)
    cs.store(group="conf/preprocessing", name="default_prep", node=PreprocessingConfig)
    cs.store(group="conf/inference", name="default_inference", node=GeneralConfig)
    cs.store(group="conf/logger", name="mlflow_logger", node=MLflowLoggerConfig)

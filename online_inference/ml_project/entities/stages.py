from dataclasses import dataclass
from typing import Optional


@dataclass
class DefaultInferenceConfig:
    run_name: Optional[str] = None
    prediction_path: str = "predict.csv"

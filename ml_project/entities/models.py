from dataclasses import dataclass
from typing import Optional


@dataclass
class LogregConfig:
    _target_: str = "sklearn.linear_model.LogisticRegression"
    penalty: str = "l1"
    solver: str = "liblinear"
    C: float = 1.0
    random_state: int = 42
    max_iter: int = 1000


@dataclass
class RfConfig:
    _target_: str = "sklearn.ensemble.RandomForestClassifier"
    n_estimators: int = 500
    criterion: str = "entropy"
    max_depth: int = 5
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = -1
    random_state: int = 42
    class_weight: Optional[str] = None

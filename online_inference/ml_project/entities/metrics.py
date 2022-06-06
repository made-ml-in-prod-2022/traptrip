from dataclasses import dataclass


@dataclass
class AccuracyConfig:
    _target_: str = "sklearn.metrics.accuracy_score"


@dataclass
class F1scoreConfig:
    _target_: str = "sklearn.metrics.f1_score"
    average: str = "binary"

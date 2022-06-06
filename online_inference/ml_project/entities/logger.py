from dataclasses import dataclass


@dataclass
class MLflowLoggerConfig:
    _target_: str = "ml_project.logger.mlflow.MLflowLogger"
    experiment_name: str = "Classification"
    run_name: str = "${general.run_name}"
    tracking_uri: str = "${general.project_dir}/mlruns"

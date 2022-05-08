import sys
import logging

sys.path.append(".")

import hydra

from ml_project.entities import Config, register_configs
from ml_project.stages.train_stage import (
    get_data,
    save_artifacts,
    train_model,
    get_score,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")
register_configs()


def train_pipeline(cfg: Config) -> None:
    # Data preprocessing
    train_data, test_data, train_target, test_target = get_data(cfg)

    # Train
    model = train_model(cfg, train_data, train_target)

    # Validation
    score = get_score(cfg, model, test_data, test_target)

    # Save artifacts
    save_artifacts(cfg, model, score)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: Config):
    train_pipeline(cfg)


if __name__ == "__main__":
    run()

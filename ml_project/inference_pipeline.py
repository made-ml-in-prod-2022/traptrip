import sys
import logging

sys.path.append(".")

import hydra

from ml_project.entities import Config, register_configs
from ml_project.stages.inference_stage import (
    get_checkpoint_path,
    load_model,
    get_data,
    make_prediction,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")
register_configs()


def inference_pipeline(cfg: Config):
    # Check checkpoint path
    checkpoint_path = get_checkpoint_path(cfg)

    # Load the model
    model = load_model(checkpoint_path)

    # Data preprocessing
    data = get_data(cfg.dataset)

    # Predict & Save
    make_prediction(model, data, cfg.inference.prediction_path)

    logging.info(f"Prediction saved to {cfg.inference.prediction_path}")


@hydra.main(config_path="conf", config_name="config")
def run(cfg: Config):
    inference_pipeline(cfg)


if __name__ == "__main__":
    run()

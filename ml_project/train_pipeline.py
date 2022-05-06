import os
import pickle
import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf

from utils.utils import load_obj
from preprocessing.dataset import Dataset


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")


@hydra.main(config_path="conf", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data, target = Dataset(cfg.dataset).load_dataset()
    transformer = instantiate(cfg.preprocessing)
    data = transformer.fit_transform(data, target)
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=cfg.dataset.test_size, random_state=cfg.general.seed
    )

    model = instantiate(cfg.model)
    model.fit(train_data, train_target)

    metric = load_obj(cfg.metric._target_)
    score = metric(test_target, model.predict(test_data))
    logging.info(f"{metric.__name__}: {score:.6f}")

    # Save artifacts
    Path(cfg.general.checkpoint_path).mkdir(exist_ok=True)
    model_path = os.path.join(cfg.general.checkpoint_path, "model.pkl")
    transformer_path = os.path.join(cfg.general.checkpoint_path, "data_transformer.pkl")
    with open(model_path, "wb") as fout:
        pickle.dump(model, fout)
    with open(transformer_path, "wb") as fout:
        pickle.dump(transformer, fout)
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Data transformer saved to {transformer_path}")


if __name__ == "__main__":
    run()

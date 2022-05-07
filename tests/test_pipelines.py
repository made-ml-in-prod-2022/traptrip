import os
import pytest
from ml_project.train_pipeline import train_pipeline
from ml_project.inference_pipeline import inference_pipeline


def test_train_pipeline(tmp_path, config, fake_dataset_path):
    checkpoint_path = str(tmp_path.joinpath(config.general.checkpoint_path))
    model_path = os.path.join(checkpoint_path, "model.pkl")
    transformer_path = os.path.join(checkpoint_path, "data_transformer.pkl")

    config.dataset.datapath = fake_dataset_path
    config.general.checkpoint_path = checkpoint_path
    train_pipeline(config)

    assert os.path.exists(model_path), "Model isn't saved!"
    assert os.path.exists(transformer_path), "Data transformer isn't saved!"


@pytest.fixture
def artifacts_path(tmp_path, config, fake_dataset_path):
    checkpoint_path = str(tmp_path.joinpath(config.general.checkpoint_path))
    config.dataset.datapath = fake_dataset_path
    config.general.checkpoint_path = checkpoint_path
    train_pipeline(config)
    return checkpoint_path


def test_inference_pipeline(
    tmp_path, artifacts_path, config, fake_dataset, fake_dataset_path
):
    prediction_path = str(tmp_path.joinpath(config.inference.prediction_path))
    config.dataset.datapath = fake_dataset_path
    config.general.checkpoint_path = artifacts_path
    config.inference.prediction_path = prediction_path

    inference_pipeline(config)

    assert os.path.exists(prediction_path), "There is no prediction file!"

    with open(prediction_path, "r") as fin:
        lines = fin.readlines()

    assert len(lines) == len(
        fake_dataset
    ), f"Prediction size ({len(lines)}) is not equal with dataset size ({len(fake_dataset)})"

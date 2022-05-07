import os

from ml_project.train_pipeline import train_pipeline


def test_train_pipeline(tmp_path, config, fake_dataset_path):
    checkpoint_path = str(tmp_path.joinpath(config.general.checkpoint_path))
    model_path = os.path.join(checkpoint_path, "model.pkl")
    transformer_path = os.path.join(checkpoint_path, "data_transformer.pkl")

    config.dataset.datapath = fake_dataset_path
    config.general.checkpoint_path = checkpoint_path
    train_pipeline(config)

    assert os.path.exists(model_path), "Model isn't saved!"
    assert os.path.exists(transformer_path), "Data transformer isn't saved!"


def test_inference_pipeline(tmp_path, config, fake_dataset_path):
    pass

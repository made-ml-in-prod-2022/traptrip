import numpy as np
from ml_project.preprocessing.dataset import Dataset
from ml_project.preprocessing.data_transformer import DefaultTransformer


def test_data_transformer(config, fake_dataset):
    transformer = DefaultTransformer(
        config.preprocessing.numerical, config.preprocessing.categorial
    )
    data = fake_dataset.drop(config.dataset.target_column, axis=1)
    target = fake_dataset[config.dataset.target_column]

    data = transformer.fit_transform(data, target)

    print(data)
    assert np.allclose(
        data[:, : len(config.preprocessing.numerical)].mean(axis=0), 0, atol=1e-6
    ) and np.allclose(
        data[:, : len(config.preprocessing.numerical)].std(axis=0), 1, atol=1e-6
    ), "Num features were not standartized"


def test_dataset_loading(config, fake_dataset, fake_dataset_path):
    config.dataset.datapath = fake_dataset_path
    data, target = Dataset(config.dataset).load_dataset()

    assert len(data) == len(
        fake_dataset
    ), f"dataset length should be {len(fake_dataset)} (get: {len(data)})"
    assert np.all(
        target == fake_dataset[config.dataset.target_column]
    ), "targets are different from ground truth"

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from omegaconf import DictConfig


class DefaultTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features: DictConfig):
        self.features = features
        self.transformer = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), list(features.numerical)),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(features.categorial),
                ),
            ]
        )
        self.mean = None
        self.std = None

    def fit(self, data: pd.DataFrame, target=None):
        self.mean = data[self.features.numerical].mean(axis=0)
        self.std = data[self.features.numerical].std(axis=0)
        self.transformer.fit(data, target)
        return self

    def transform(self, data, target=None):
        data.loc[:, self.features.numerical] = (
            data.loc[:, self.features.numerical] - self.mean
        ) / self.std
        data = self.transformer.transform(data)
        return data

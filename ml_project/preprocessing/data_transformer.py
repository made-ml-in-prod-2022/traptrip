from typing import Any, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class DefaultTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical: List[Any], categorial: List[Any]):
        self.transformer = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), list(numerical)),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(categorial),
                ),
            ]
        )

    def fit(self, data: pd.DataFrame, target=None):
        self.transformer.fit(data, target)
        return self

    def transform(self, data, target=None):
        data = self.transformer.transform(data)
        return data

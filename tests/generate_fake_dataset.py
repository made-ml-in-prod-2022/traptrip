import pandas as pd
from faker import Faker
from numpy.random import normal


def generate_dataset(n_rows: int, real_df_path: str):
    fake = Faker()
    real_df = pd.read_csv(real_df_path)
    fake_data = dict()

    for col in real_df.columns:
        if real_df[col].nunique() < 100:
            fake_data[col] = fake.random_elements(
                elements=real_df[col].unique(), length=n_rows
            )
        else:
            fake_data[col] = [
                normal(real_df[col].mean(), real_df[col].std()) for _ in range(n_rows)
            ]

    fake_df = pd.DataFrame(fake_data)
    return fake_df

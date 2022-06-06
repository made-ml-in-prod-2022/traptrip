"""Script to check fastAPI server work correctly"""

import logging

import requests
import pandas as pd
from faker import Faker
from numpy.random import normal

from transport.settings import CONFIG

N_ROWS = 10
URL = f"http://{CONFIG.host}:{CONFIG.port}/predict"
logger = logging.getLogger(__name__)


def generate_dataset(n_rows: int):
    fake = Faker()
    fake_data = {
        "sex": fake.random_elements(elements=list(range(2)), length=n_rows),
        "cp": fake.random_elements(elements=list(range(4)), length=n_rows),
        "fbs": fake.random_elements(elements=list(range(2)), length=n_rows),
        "restecg": fake.random_elements(elements=list(range(3)), length=n_rows),
        "exang": fake.random_elements(elements=list(range(2)), length=n_rows),
        "slope": fake.random_elements(elements=list(range(3)), length=n_rows),
        "ca": fake.random_elements(elements=list(range(4)), length=n_rows),
        "thal": fake.random_elements(elements=list(range(3)), length=n_rows),
        "age": [normal(54.54, 9.05) for _ in range(n_rows)],
        "trestbps": [normal(131.69, 17.76) for _ in range(n_rows)],
        "chol": [normal(247.35, 52) for _ in range(n_rows)],
        "thalach": [normal(149.6, 22.94) for _ in range(n_rows)],
        "oldpeak": [normal(1.06, 1.17) for _ in range(n_rows)],
    }
    fake_df = pd.DataFrame(fake_data)
    return fake_df


def request_predict():
    dataset = generate_dataset(N_ROWS)
    data = [row.tolist() for _, row in dataset.iterrows()]
    features = dataset.columns.tolist()

    health_response = requests.get(URL.replace("predict", "healthcheck"))
    print(health_response.status_code, health_response.content)

    response = requests.post(URL, json={"data": data, "features": features})
    print(f"Status: {response.status_code} ; Response data: {response.json()}")


if __name__ == "__main__":
    request_predict()

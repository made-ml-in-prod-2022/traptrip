import os
from pathlib import Path
from datetime import timedelta

PROJECT_DIR = Path("/Users/and/projects/MADE/2sem/traptrip/airflow_ml_dags")
DEFAULT_AIRFLOW_ARGS = {
    "owner": "airflow",
    "email_on_failure": True,
    "email": ["aletni@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
MLFLOW_PARAMS = {"MLFLOW_TRACKING_URL": os.environ.get("MLFLOW_TRACKING_URL", "http://localhost:5001")}

DATA_VOLUME = str(PROJECT_DIR / "data" / ":/data")
ARTIFACTS_VOLUME = str(PROJECT_DIR / "mlflow_logs" / ":/mlruns")

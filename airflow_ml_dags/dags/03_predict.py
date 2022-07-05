from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from settings import DEFAULT_AIRFLOW_ARGS, DATA_VOLUME, MLFLOW_PARAMS, ARTIFACTS_VOLUME


with DAG(
    dag_id="03_predict",
    description="Inference model",
    schedule_interval="@daily",
    start_date=days_ago(3),
    default_args=DEFAULT_AIRFLOW_ARGS,
) as dag:

    start_task = DummyOperator(task_id="start-inference")

    features_sensor = FileSensor(
        task_id="sensor-features",
        poke_interval=10,
        retries=5,
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv",        
    )

    prediction = DockerOperator(
        task_id="predict",
        image="airflow-predict",
        command="predict.py --input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }} "
                "--data-transformers-dir /data/data-transformers",
        network_mode="host",
        do_xcom_push=False,
        private_environment=MLFLOW_PARAMS,
        volumes=[DATA_VOLUME, ARTIFACTS_VOLUME],
    )

    end_task = DummyOperator(task_id="end-inference")

    (
        start_task 
        >> features_sensor
        >> prediction 
        >> end_task
    )

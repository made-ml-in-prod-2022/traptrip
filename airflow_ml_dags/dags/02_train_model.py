from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from settings import DEFAULT_AIRFLOW_ARGS, DATA_VOLUME, ARTIFACTS_VOLUME, MLFLOW_PARAMS


with DAG(
    dag_id="02_train_model",
    description="Train & validate model",
    schedule_interval="@weekly",
    start_date=days_ago(3),
    default_args=DEFAULT_AIRFLOW_ARGS,
) as dag:

    start_task = DummyOperator(task_id="start-training")

    features_sensor = FileSensor(
        task_id="sensor-features",
        poke_interval=10,
        retries=5,
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv",        
    )

    targets_sensor = FileSensor(
        task_id="sensor-targets",
        poke_interval=10,
        retries=5,
        filepath="/opt/airflow/data/raw/{{ ds }}/target.csv",        
    )

    split_data = DockerOperator(
        task_id="split-data",
        image="airflow-split-data",
        command="split.py --input-dir /data/raw/{{ ds }} --output-dir /data/split/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DATA_VOLUME],
    )

    preprocess_data = DockerOperator(
        task_id="preprocessing",
        image="airflow-preprocess",
        command="preprocess.py --input-dir /data/split/{{ ds }} --output-dir /data/processed/{{ ds }} --save-transformer-dir /data/data-transformers/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DATA_VOLUME],
    )

    train_model = DockerOperator(
        task_id="train-model",
        image="airflow-train",
        command="train.py --data-dir /data/processed/{{ ds }} --save-model-dir /data/models/{{ ds }}",
        network_mode="host",
        do_xcom_push=False,
        private_environment=MLFLOW_PARAMS,
        volumes=[DATA_VOLUME, ARTIFACTS_VOLUME],
    )

    validate_model = DockerOperator(
        task_id="validate-model",
        image="airflow-validate",
        command="validate.py --data-dir /data/processed/{{ ds }} --metrics-dir /data/metrics/{{ ds }}",
        network_mode="host",
        do_xcom_push=False,
        private_environment=MLFLOW_PARAMS,
        volumes=[DATA_VOLUME, ARTIFACTS_VOLUME],
    )

    end_task = DummyOperator(task_id="end-training")

    (
        start_task 
        >> [features_sensor, targets_sensor] 
        >> split_data 
        >> preprocess_data 
        >> train_model 
        >> validate_model 
        >> end_task
    )

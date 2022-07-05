from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

from settings import DEFAULT_AIRFLOW_ARGS, DATA_VOLUME


with DAG(
    dag_id="01_generate_data",
    description="Generate dataset for training",
    schedule_interval="@daily",
    start_date=days_ago(3),
    default_args=DEFAULT_AIRFLOW_ARGS,
) as dag:

    start_task = DummyOperator(task_id="start-downloading")

    download_data = DockerOperator(
        task_id="airflow-download",
        image="airflow-download",
        command="download.py /data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DATA_VOLUME],
    )

    end_task = DummyOperator(task_id="end-downloading")

    start_task >> download_data >> end_task

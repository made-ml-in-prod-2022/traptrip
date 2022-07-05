import sys

import pytest
from airflow.models import DagBag

sys.path.append("dags/")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="/testing/dags", include_examples=False)


def test_generate_data_dag_loaded(dag_bag):
    assert "01_generate_data" in dag_bag.dags
    assert len(dag_bag.dags["01_generate_data"].tasks) == 3


def test_generate_data_dag_structure(dag_bag):
    dags_structure = {
        "start-downloading": ["airflow-download"],
        "airflow-download": ["end-downloading"],
        "end-downloading": [],
    }
    dag = dag_bag.dags["01_generate_data"]
    for name, task in dag.task_dict.items():
        assert set(dags_structure[name]) == task.downstream_task_ids


def test_train_model_dag_loaded(dag_bag):
    assert "02_train_model" in dag_bag.dags
    assert len(dag_bag.dags["02_train_model"].tasks) == 8


def test_train_model_dag_structure(dag_bag):
    structure = {
        "start-training": ["sensor-features", "sensor-targets"],
        "sensor-features": ["split-data"],
        "sensor-targets": ["split-data"],
        "split-data": ["preprocessing"],
        "preprocessing": ["train-model"],
        "train-model": ["validate-model"],
        "validate-model": ["end-training"],
        "end-training": [],
    }
    dag = dag_bag.dags["02_train_model"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_predictions_dag_loaded(dag_bag):
    assert "03_predict" in dag_bag.dags
    assert len(dag_bag.dags["03_predict"].tasks) == 4


def test_predictions_dag_structure(dag_bag):
    structure = {
        "start-inference": ["sensor-features"],
        "sensor-features": ["predict"],
        "predict": ["end-inference"],
        "end-inference": [],
    }
    dag = dag_bag.dags["03_predict"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids

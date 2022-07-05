ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

WORKDIR /testing

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==1.2.0

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt --no-cache-dir && \
    pip install --upgrade protobuf==3.20.0

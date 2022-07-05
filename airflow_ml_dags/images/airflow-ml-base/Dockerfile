FROM python:3.8-slim

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt --no-cache-dir && \
    pip install --upgrade protobuf==3.20.0 && \
    rm /requirements.txt

# [Homework 2] Online Inference

## Python version 
Python 3.9.10

Dockerized FastAPI server for heart desease recognition.

## Start Server locally
```bash
source .env
uvicorn transport.app:app --port $PORT
```

## Start in docker
**Build locally**

```bash
source .env
docker build -t pythondestroyer/heart_desease_server:v2 .
docker pull pythondestroyer/heart_desease_server:v2
docker run -p $PORT:$PORT pythondestroyer/heart_desease_server:v2
```

**Pull from docker hub**

```bash
source .env
docker pull pythondestroyer/heart_desease_server:v2
docker run -p $PORT:$PORT pythondestroyer/heart_desease_server:v2
```

## Check server with requests
```bash
python generate_request.py
```

Project Organization
------------
    ├── README.md                   <- The top-level README for developers using this project.
    ├── model                       <- folder with pretrained model file.
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    ├── requirements-dev.txt        <- The requirements with packages for development & testing
    ├── config.yaml                 <- configuration file.
    ├── .env-example                <- Example of .env file.
    ├── generate_request.py         <- Script for server work checking.
    ├── Dockerfile                  <- File to build a docker image.
    └── transport                   <- Dataclasses for config validation
        ├── app.py                  <- FastAPI server logic.
        ├── schemas.py              <- Validation schemas.
        ├── settings.py             <- main settings of an application.
        └── utils.py                <- Useful functions for an application.
--------

## Docker optimizations
1. Use python:3.9-slim instead of python:3.9
2. Add files useless for container to .dockerignore
Final image size is 149.45 Mb (compressed).
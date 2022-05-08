# ML in prod
Production ready project to solve classification task from Kaggle Dataset "[Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)"

## Python version 
Python 3.9.10

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Train
```bash
python ml_project/train_pipeline.py
```
Also you can change configurations when run the script. For example:
```bash
python ml_project/train_pipeline.py model=rf
```
Or use multirun option:
```bash
python ml_project/train_pipeline.py model=logreg,rf metric=accuracy,f1_score --multyrun
```
You will get output like this
```bash
[2022-05-07 13:47:09,743][HYDRA] Launching 4 jobs locally
[2022-05-07 13:47:09,743][HYDRA]        #0 : model=logreg metric=accuracy
[2022-05-07 13:47:09,822][root][INFO] - accuracy_score: 0.833333
[2022-05-07 13:47:09,823][root][INFO] - Model saved to weights/model.pkl
[2022-05-07 13:47:09,823][root][INFO] - Data transformer saved to weights/data_transformer.pkl
[2022-05-07 13:47:09,823][HYDRA]        #1 : model=logreg metric=f1_score
[2022-05-07 13:47:09,913][root][INFO] - f1_score: 0.827586
[2022-05-07 13:47:09,914][root][INFO] - Model saved to weights/model.pkl
[2022-05-07 13:47:09,914][root][INFO] - Data transformer saved to weights/data_transformer.pkl
[2022-05-07 13:47:09,915][HYDRA]        #2 : model=rf metric=accuracy
[2022-05-07 13:47:10,398][root][INFO] - accuracy_score: 0.766667
[2022-05-07 13:47:10,407][root][INFO] - Model saved to weights/model.pkl
[2022-05-07 13:47:10,407][root][INFO] - Data transformer saved to weights/data_transformer.pkl
[2022-05-07 13:47:10,408][HYDRA]        #3 : model=rf metric=f1_score
[2022-05-07 13:47:10,896][root][INFO] - f1_score: 0.766667
[2022-05-07 13:47:10,903][root][INFO] - Model saved to weights/model.pkl
[2022-05-07 13:47:10,903][root][INFO] - Data transformer saved to weights/data_transformer.pkl
```
Results will be in config.general.artifacts_path/<current_datetime>

### Inference
```bash
python ml_project/inference_pipeline.py \ 
    'dataset.datapath=${dataset.data_dir}/test_data_file_name.csv' \ 
    'inference.run_name=folder_with_artifacts'
```
Example:
```bash
python ml_project/inference_pipeline.py \ 
    'dataset.datapath=${dataset.data_dir}/test.csv' \ 
    'inference.run_name=2022-05-08_12-52-48'
```
Results will be in config.general.artifacts_path/<current_datetime>
### Tests
**Install test requirements**
```bash
pip install -r requirements-dev.txt
```

**Run**
```bash
pytest
```

Project Organization
------------
    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data                        <- The data sets for modeling.
    ├── notebooks                   <- Jupyter notebooks
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    ├── requirements-dev.txt        <- The requirements with packages for development & testing
    │
    ├── ml_project                  <- Source code for use in this project.
    │   ├── conf                    <- Configuration files
    │   │   ├── dataset         
    │   │   ├── preprocessing   
    │   │   ├── metric         
    │   │   ├── model           
    │   │   └── config.yaml
    │   │ 
    │   ├── preprocessing           <- Data preprocessing scripts
    │   │   ├── data_transformer.py 
    │   │   └── dataset.py
    │   │
    │   ├── utils                   <- Helpful functions for pipeline
    │   │   └── technical_utils.py
    │   │
    │   ├── entities                <- Dataclasses for config validation
    │   │   ├── config.py
    │   │   ├── dataset.py
    │   │   ├── metrics.py
    │   │   └── models.py
    │   │
    │   ├── train_pipeline.py       <- Train pipeline main script
    │   └── inference_pipeline.py   <- Inference pipeline main script
    │
    ├── tests                       <- Tests for pipelines and functions
    ├── setup.cfg                   <- Store pytest configurations
    └── setup.py                    <- Makes project pip installable (pip install -e .) so src can be imported
--------

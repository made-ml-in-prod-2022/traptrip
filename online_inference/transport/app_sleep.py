import os
import logging
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response

from .settings import CONFIG
from .utils import load_pkl
from .schemas import HeardDeseaseRequest, HeardDeseaseResponse

START_SLEEP = 10
TIME_TO_LIVE = 60
START_TIME = None

logger = logging.getLogger(__name__)
logger.setLevel(CONFIG.log_level)
model = None
app = FastAPI()


@app.on_event("startup")
async def load_model():
    global model, START_TIME
    time.sleep(START_SLEEP)
    if not os.path.exists(CONFIG.model_path):
        raise FileNotFoundError("There is no model in this path: '{CONFIG.model_path}'")
    model = load_pkl(Path(CONFIG.model_path))
    logger.info("Model loaded!")
    START_TIME = time.time()

@app.get("/healthcheck")
async def healthcheck():
    """Check if model loaded correctly"""
    if model is None:
        raise HTTPException(404, "Model not found!")
    
    cur_time = time.time()
    if cur_time - START_TIME > TIME_TO_LIVE:
        raise Exception("App is stopped!")
    
    return Response("Model loaded correctly!")


@app.post("/predict", response_model=HeardDeseaseResponse)
async def predict(request: HeardDeseaseRequest):
    """Post a file to recognize"""

    df = pd.DataFrame(request.data, columns=request.features)
    prediction = model.predict_proba(df)[:, 1].tolist()
    return HeardDeseaseResponse(result=prediction)

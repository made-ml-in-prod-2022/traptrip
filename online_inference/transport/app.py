import os
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response

from .settings import CONFIG
from .utils import load_pkl
from .schemas import HeardDeseaseRequest, HeardDeseaseResponse

logger = logging.getLogger(__name__)
logger.setLevel(CONFIG.log_level)
model = None
app = FastAPI()


@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(CONFIG.model_path):
        raise FileNotFoundError(f"There is no model in this path: '{CONFIG.model_path}'")
    model = load_pkl(Path(CONFIG.model_path))
    logger.info("Model loaded!")


@app.get("/healthcheck")
async def healthcheck():
    """Check if model loaded correctly"""
    if model is None:
        raise HTTPException(404, "Model not found!")
    return Response("Model loaded correctly!")


@app.post("/predict", response_model=HeardDeseaseResponse)
async def predict(request: HeardDeseaseRequest):
    """Post a file to recognize"""

    df = pd.DataFrame(request.data, columns=request.features)
    prediction = model.predict_proba(df)[:, 1].tolist()
    return HeardDeseaseResponse(result=prediction)

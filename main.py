# main.py

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Load the ARIMA model
model = joblib.load("./arima_model.joblib")


@app.get("/predict")
def predict():
    try:
        # Make prediction using the loaded ARIMA model
        prediction = model.forecast(steps=4)  # Adjust as needed
        return {"prediction": prediction.iloc[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

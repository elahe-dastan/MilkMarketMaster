# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Load the ARIMA model
model = joblib.load('./arima_model.joblib')


class PredictionInput(BaseModel):
    value: float


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Make prediction using the loaded ARIMA model
        prediction = model.forecast(steps=4)  # Adjust as needed
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

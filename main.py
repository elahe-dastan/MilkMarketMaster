# main.py

import datetime
import logging
from typing import Annotated

import joblib
import pandas as pd
import pydantic
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


class Response(pydantic.BaseModel):
    buy_at: datetime.date
    max_profit: float
    sell_at: datetime.date


def load_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


def maximize_profit(forecast_df):
    # Find the index of the maximum value
    max_index = forecast_df["forecast_price"].idxmax()

    # Find the index of the minimum value before the maximum value
    min_index_before_max = forecast_df["forecast_price"].loc[:max_index].idxmin()

    # Calculate the maximum profit
    max_profit = (
        forecast_df["forecast_price"][max_index]
        - forecast_df["forecast_price"][min_index_before_max]
    )

    response = Response(
        buy_at=forecast_df["forecast_date"][min_index_before_max].date(),
        sell_at=forecast_df["forecast_date"][max_index].date(),
        max_profit=max_profit,
    )

    # Print the result
    logger.info(
        f"Buy at index {response.buy_at}, sell at index {response.sell_at}, max profit: {response.max_profit}"
    )

    return response


@app.get("/predict")
def predict(
    country_id: Annotated[
        int, Query(title="Country ID", description="ID of the country", gt=0)
    ],
    product_id: Annotated[
        int, Query(title="Product ID", description="ID of the product", gt=0)
    ],
):
    try:
        logger.info("hi")
        # Make prediction using the loaded ARIMA model
        # steps = 16 means 4 months
        model = load_model(country_id, product_id)
        forecasts = model.forecast(steps=16)  # Adjust as needed
        last_date = pd.to_datetime("2023-12-21")
        logger.info("hi")
        forecast_df = pd.DataFrame(
            {
                "forecast_date": pd.date_range(
                    start=last_date + pd.DateOffset(1), periods=16, freq="W-THU"
                ),
                "forecast_price": forecasts,
            }
        )
        # forecast_df.set_index('forecast_date', inplace=True)

        return maximize_profit(forecast_df)
    except Exception as e:
        logger.exception("something bad happend")
        raise HTTPException(status_code=500, detail=str(e))

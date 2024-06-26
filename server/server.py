import datetime
import logging
from typing import Annotated

import joblib
import pandas as pd
import pydantic
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()
logger = logging.getLogger("main")


class Response(pydantic.BaseModel):
    buy_at: datetime.date
    max_profit: float
    sell_at: datetime.date


def load_model(country_id: int, product_id: int, name: str):
    path = f"./data/{country_id}/{product_id}/{name}.joblib"

    try:
        model = joblib.load(path)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model is not available")

    return model


def load_inflation(country_id) -> pd.DataFrame:
    path = f"./data/{country_id}/inflation_rates.csv"
    # Load the inflation csv
    # HANDLE EXCEPTION **********************
    inflation: pd.DataFrame = pd.read_csv(path)
    # FOR NOW --> Quarterly
    yearly_inflation = inflation[inflation["data_interval"] == "yearly"]
    assert isinstance(yearly_inflation, pd.DataFrame)
    inflation = yearly_inflation
    inflation = inflation.sort_values(by="date")

    inflation["date"] = pd.to_datetime(inflation["date"])
    inflation.set_index("date", inplace=True)

    selected_inflation = inflation[["rate"]]
    assert isinstance(selected_inflation, pd.DataFrame)

    return selected_inflation


def merge_forecast_inflation(forecast_df: pd.DataFrame, inflation: pd.DataFrame):
    forecast_df.index = pd.to_datetime(forecast_df.index)
    forecast_df["year"] = forecast_df.index.year  # type: ignore
    forecast_df["date"] = forecast_df.index

    inflation.index = pd.to_datetime(inflation.index)
    inflation["year"] = inflation.index.year  # type: ignore

    return pd.merge(
        forecast_df, inflation, left_on="year", right_on="year", how="inner"
    )


def maximize_profit(forecast_df):
    # Find the index of the maximum value
    max_index = forecast_df["adjusted_price"].idxmax()

    # Find the index of the minimum value before the maximum value
    min_index_before_max = forecast_df["adjusted_price"].loc[:max_index].idxmin()

    # Calculate the maximum profit
    max_profit = (
        forecast_df["adjusted_price"][max_index]
        - forecast_df["adjusted_price"][min_index_before_max]
    )

    response = Response(
        buy_at=min_index_before_max,
        sell_at=max_index,
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
    steps: Annotated[
        int,
        Query(title="Steps", description="Number of predicted steps", gt=0),
    ] = 16,
    df: Annotated[
        bool,
        Query(
            title="Dataframe",
            description="Return the whole predicted dataframe",
        ),
    ] = False,
    param: Annotated[
        str,
        Query(
            title="Parameter",
            description="The parameter that is going to predict",
        ),
    ] = "price",
):
    try:
        production_model = load_model(country_id, product_id, "production_arima_model")
        production_forecasts = pd.DataFrame(
            production_model.forecast(steps=steps)
        ).rename(columns={"predicted_mean": "production_value"})
        production_forecasts["year"] = production_forecasts.index.year  # type: ignore
        production_forecasts["month"] = production_forecasts.index.month  # type: ignore
        logger.info(production_forecasts)

        smp_model = load_model(country_id, product_id, "smp_quotations_arima_model")
        smp_forecasts = pd.DataFrame(smp_model.forecast(steps=steps)).rename(
            columns={"predicted_mean": "estimated_price"}
        )
        smp_forecasts["year"] = smp_forecasts.index.year  # type: ignore
        smp_forecasts["month"] = smp_forecasts.index.month  # type: ignore
        logger.info(smp_forecasts)

        dataset = pd.merge(
            smp_forecasts,
            production_forecasts,
            left_on=["year", "month"],
            right_on=["year", "month"],
            how="inner",
        )
        logger.info("merge smp and productions forecasts %s", dataset.head())

        rf_model = load_model(country_id, product_id, "rf_model")
        rf_predictions = rf_model.predict(
            dataset[["estimated_price", "production_value"]]
        )
        logger.info(dataset.index)
        rf_based_forecast_df = pd.DataFrame(rf_predictions, columns=["predicted_mean"]).set_index(  # type: ignore
            smp_forecasts.index
        )
        logger.info(rf_based_forecast_df)
        logger.info(rf_based_forecast_df.columns)

        inflation = load_inflation(country_id)
        rf_based_forecast_inflation_merged = merge_forecast_inflation(
            rf_based_forecast_df, inflation
        )
        rf_based_forecast_inflation_merged.set_index("date", inplace=True)

        rf_based_forecast_inflation_merged[
            "adjusted_price"
        ] = rf_based_forecast_inflation_merged["predicted_mean"] + (
            rf_based_forecast_inflation_merged["predicted_mean"]
            * rf_based_forecast_inflation_merged["rate"]
            / 100
        )

        logger.info(
            "adjusted rf based predction using inflation %s",
            rf_based_forecast_inflation_merged.head(),
        )

        match param:
            case "production":
                if df:
                    return production_forecasts.to_dict()

                return {}

            case "price":
                forecast_inflation_merged = merge_forecast_inflation(
                    smp_forecasts, inflation
                )
                forecast_inflation_merged.set_index("date", inplace=True)

                forecast_inflation_merged["adjusted_price"] = forecast_inflation_merged[
                    "estimated_price"
                ] + (
                    forecast_inflation_merged["estimated_price"]
                    * forecast_inflation_merged["rate"]
                    / 100
                )
                # merge rf-based and non-rf prices into the response.
                forecast_inflation_merged["rf_based_adjusted_price"] = (
                    rf_based_forecast_inflation_merged["adjusted_price"]
                )

                if df:
                    return forecast_inflation_merged.to_dict()

                return maximize_profit(forecast_inflation_merged)

    except Exception as e:
        logger.exception("something bad happend")
        raise HTTPException(status_code=500, detail=str(e))

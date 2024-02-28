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


def load_model(country_id, product_id, name):
    path = f"./data/{country_id}/{product_id}/{name}.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

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
        buy_at=forecast_df.index[min_index_before_max].date(),
        sell_at=forecast_df.index[max_index].date(),
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
        logger.info(dataset)

        rf_model = load_model(country_id, product_id, "rf_model")
        predictions = rf_model.predict(dataset[["estimated_price", "production_value"]])
        logger.info(dataset.index)
        forecast_df = pd.DataFrame(predictions, columns=["predicted_mean"]).set_index(
            smp_forecasts.index
        )
        logger.info(forecast_df)
        logger.info(forecast_df.columns)

        inflation = load_inflation(country_id)
        forecast_inflation_merged = merge_forecast_inflation(forecast_df, inflation)
        forecast_inflation_merged.set_index("date", inplace=True)

        forecast_inflation_merged["adjusted_price"] = forecast_inflation_merged[
            "predicted_mean"
        ] + (
            forecast_inflation_merged["predicted_mean"]
            * forecast_inflation_merged["rate"]
            / 100
        )

        logger.info(forecast_inflation_merged)

        match param:
            case "production":
                model = load_model(country_id, product_id, "production_arima_model")
                forecasts = model.forecast(steps=steps)

                if df:
                    return forecasts.to_dict()

                return {}

            case "price":
                # Make prediction using the loaded ARIMA model
                # steps = 16 means 4 months
                model = load_model(country_id, product_id, "smp_quotations_arima_model")
                forecasts = pd.DataFrame(model.forecast(steps=steps))

                inflation = load_inflation(country_id)
                forecast_inflation_merged = merge_forecast_inflation(
                    forecasts, inflation
                )
                forecast_inflation_merged.set_index("date", inplace=True)

                forecast_inflation_merged["adjusted_price"] = forecast_inflation_merged[
                    "predicted_mean"
                ] + (
                    forecast_inflation_merged["predicted_mean"]
                    * forecast_inflation_merged["rate"]
                    / 100
                )

                if df:
                    return forecast_inflation_merged.to_dict()

                return maximize_profit(forecast_inflation_merged)

    except Exception as e:
        logger.exception("something bad happend")
        raise HTTPException(status_code=500, detail=str(e))

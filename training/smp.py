"""
provide a training phase for the smp model which is
the main model for predicting the smp price.
"""

import logging

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from .common import split_train_test, train_arima_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def read(country_id: int) -> pd.DataFrame:
    """
    read the smp price for the given country. the smp product id
    is 4.
    """
    smp_df = pd.read_csv(f"./data/{country_id}/4/smp_quotations.csv")

    smp_df = smp_df.sort_values(by="date")

    smp_df["date"] = pd.to_datetime(smp_df["date"])
    smp_df = smp_df.drop_duplicates(["date"])
    smp_df.set_index("date", inplace=True)

    # force date to have a 7 days frequency.
    smp_df = smp_df.asfreq("7D", method="pad")
    smp_df.to_csv("./data/2/4/smp_quotations_revised.csv")

    return smp_df


def evaluate_model(model, test_df: pd.DataFrame):
    forecasts = model.forecast(len(test_df))
    forecasts = pd.DataFrame(forecasts)
    forecasts["date"] = forecasts.index

    inflation: pd.DataFrame = pd.read_csv("./data/2/inflation_rates.csv")
    # drop quarterly inflation rate to use yearly only
    yearly_inflation = inflation[inflation["data_interval"] == "yearly"]
    # these asserts are here to be compatible with type checking
    # system
    assert isinstance(yearly_inflation, pd.DataFrame)
    inflation = yearly_inflation
    inflation = inflation.sort_values(by="date")

    inflation["date"] = pd.to_datetime(inflation["date"])
    inflation.set_index("date", inplace=True)

    forecasts.index = pd.to_datetime(forecasts.index)
    inflation.index = pd.to_datetime(inflation.index)
    forecasts["year"] = forecasts.index.year  # type: ignore
    inflation["year"] = inflation.index.year  # type: ignore

    merged_df = pd.merge(
        forecasts, inflation, left_on="year", right_on="year", how="inner"
    )
    merged_df.set_index("date", inplace=True)

    forecasts["adjusted_price"] = merged_df["predicted_mean"] * (
        1 + (merged_df["rate"] / 100)
    )

    merged_df = merged_df.drop(columns=["year"])

    forecasts_mse = mean_squared_error(test_df["price"], forecasts["adjusted_price"])
    print(f"Forecast Mean Squared Error: {forecasts_mse}")


def smp(country_id: int, do_evaluate: bool = True, do_write: bool = True) -> ARIMA:
    """
    read the data csv and then train an ARIMA model
    for the smp. using `do_evaludate` and `do_wite`
    you can control the write and the evaluation phases.
    """
    data = read(country_id)
    train, test = split_train_test(data)

    model = train_arima_model(train, 12, 2, 1)

    if do_evaluate:
        evaluate_model(model, test)

    if do_write:
        filename = f"./data/{country_id}/4/smp_quotations_arima_model.joblib"
        joblib.dump(model, filename)

    return model

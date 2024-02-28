"""
provide a training phase for the production model which is
the main model for predicting the smp production value.
"""

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from .common import split_train_test, train_arima_model


def read(country_id: int):
    production_2_4 = pd.read_csv(f"./data/{country_id}/4/production.csv")

    production_2_4 = production_2_4.sort_values(by="date")

    production_2_4["date"] = pd.to_datetime(production_2_4["date"])
    production_2_4.set_index("date", inplace=True)

    return production_2_4


def evaluate_model(fitted_model, test_dataset, train_dateset):
    predictions = fitted_model.predict(
        start=len(train_dateset), end=len(train_dateset) + len(test_dataset) - 1
    )
    forecasts = fitted_model.forecast(len(test_dataset))
    mse = mean_squared_error(test_dataset["value"], predictions)
    forecasts_mse = mean_squared_error(test_dataset["value"], forecasts)
    print(f"Mean Squared Error: {mse}")
    print(f"Forecast Mean Squared Error: {forecasts_mse}")


def production(
    country_id: int, do_evaluate: bool = True, do_write: bool = True
) -> ARIMA:
    """
    read the data csv and then train an ARIMA model
    for the smp production value. using `do_evaludate` and `do_wite`
    you can control the write and the evaluation phases.
    """

    data = read(country_id)
    train, test = split_train_test(data)

    model = train_arima_model(train, 16, 0, 5)

    if do_evaluate:
        evaluate_model(model, test, train)

    if do_write:
        filename = f"./data/{country_id}/4/production_arima_model.joblib"
        joblib.dump(model, filename)

    return model

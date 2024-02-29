import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .common import split_train_test


def load_smp_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    model = joblib.load(path)

    return model


def load_production_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/production_arima_model.joblib"
    model = joblib.load(path)

    return model


def read_smp_data():
    smp_2_4 = pd.read_csv("./data/2/4/smp_quotations_revised.csv")
    smp_2_4 = smp_2_4.sort_values(by="date")
    smp_2_4["date"] = pd.to_datetime(smp_2_4["date"])
    return smp_2_4


def prepare_data(country_id: int):
    smp_model = load_smp_model(country_id, 4)
    smp_predictions = pd.DataFrame(smp_model.predict(start=0, end=3000))
    smp_predictions["date"] = smp_predictions.index
    smp_predictions["year"] = smp_predictions.index.year  # type: ignore
    smp_predictions["month"] = smp_predictions.index.month  # type: ignore
    smp_predictions = smp_predictions.rename(
        columns={"predicted_mean": "estimated_price"}
    )

    production_model = load_production_model(country_id, 4)
    production_predictions = pd.DataFrame(production_model.predict(start=0, end=3000))
    production_predictions["year"] = production_predictions.index.year  # type: ignore
    production_predictions["month"] = production_predictions.index.month  # type: ignore
    production_predictions = production_predictions.rename(
        columns={"predicted_mean": "production_value"}
    )

    return smp_predictions, production_predictions


def feature_engineering(smp_predictions, production_predictions):
    smp_data = read_smp_data()
    smp = pd.merge(
        smp_predictions, smp_data, left_on="date", right_on="date", how="inner"
    )

    smp = smp[["estimated_price", "price", "year", "month"]]

    dataset = pd.merge(
        smp,
        production_predictions,
        left_on=["year", "month"],
        right_on=["year", "month"],
        how="inner",
    )

    return dataset


def train_model(train) -> RandomForestRegressor:
    regr = RandomForestRegressor(max_depth=4)
    regr.fit(train[["estimated_price", "production_value"]], train["price"])

    return regr


def evaluate_model(fitted_model, test_dataset):
    predictions = fitted_model.predict(
        test_dataset[["estimated_price", "production_value"]]
    )

    mse = mean_squared_error(test_dataset["price"], predictions)

    print(f"Mean Squared Error: {mse}")


def rf(
    country_id: int, do_evaluate: bool = True, do_write: bool = True
) -> RandomForestRegressor:
    """
    read the csv data and model to train a random forest model
    for considering the effect of different parameters on smp price.
    this should be trained on the last step after having models for all the parameters.
    using `do_evaludate` and `do_wite` you can control the write and the evaluation phases.
    """
    smp_predictions, production_predictions = prepare_data(country_id)
    dataset = feature_engineering(smp_predictions, production_predictions)
    train, test = split_train_test(dataset)
    model = train_model(train)

    if do_evaluate:
        evaluate_model(model, test)

    if do_write:
        filename = f"./data/{country_id}/4/rf_model.joblib"
        joblib.dump(model, filename)

    return model

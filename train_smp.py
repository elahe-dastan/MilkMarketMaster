import logging

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def read() -> pd.DataFrame:
    smp_df = pd.read_csv("./data/2/4/smp_quotations.csv")

    smp_df = smp_df.sort_values(by="date")

    smp_df["date"] = pd.to_datetime(smp_df["date"])
    smp_df = smp_df.drop_duplicates(["date"])
    smp_df.set_index("date", inplace=True)

    smp_df = smp_df.asfreq("7D", method="pad")

    return smp_df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * 0.9)
    train, test = df[:train_size], df[train_size:]
    # these asserts are here to be compatible with type checking
    # system
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    return train, test


def train_model(train_dataset, p, d, q):
    order = (p, d, q)
    model = ARIMA(train_dataset["price"], order=order)
    fitted_model = model.fit()

    return fitted_model


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


def load_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


if __name__ == "__main__":
    data = read()
    train, test = split_train_test(data)

    model = train_model(train, 12, 2, 1)

    evaluate_model(model, test)
    # model = load_model(2, 4)

    # # Save the fitted model to a file using joblib
    filename = "./data/2/4/smp_quotations_arima_model.joblib"
    joblib.dump(model, filename)

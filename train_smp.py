import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def read():
    smp_2_4 = pd.read_csv("./data/2/4/smp_quotations.csv")

    smp_2_4 = smp_2_4.sort_values(by="date")

    smp_2_4["date"] = pd.to_datetime(smp_2_4["date"])
    smp_2_4 = smp_2_4.drop_duplicates(["date"])
    smp_2_4["date_b"] = smp_2_4["date"]
    smp_2_4.set_index("date", inplace=True)
    smp_2_4 = smp_2_4.asfreq("7D", method="pad")

    smp_2_4.to_csv("./data/2/4/smp_quotations_revised.csv")

    return smp_2_4


def split(df):
    train_size = int(len(df) * 0.95)  # 80% for training, adjust as needed
    train, test = df[:train_size], df[train_size:]

    return train, test


def train_model(train_dataset, p, d, q):
    order = (p, d, q)
    model = ARIMA(train_dataset["price"], order=order)
    fitted_model = model.fit()

    return fitted_model


def evaluate_model(fitted_model, test_dataset):
    forecasts = fitted_model.forecast(32)
    forecasts = pd.DataFrame(forecasts)

    inflation = pd.read_csv("./data/2/inflation_rates.csv")
    inflation = inflation[inflation["data_interval"] == "yearly"]
    inflation = inflation.sort_values(by="date")

    inflation["date"] = pd.to_datetime(inflation["date"])
    inflation.set_index("date", inplace=True)

    # Extract the year from the date index in df_weekly
    forecasts.index = pd.to_datetime(forecasts.index)
    forecasts["year"] = forecasts.index.year
    inflation["year"] = inflation.index.year

    # Merge based on the condition: year in df_weekly should match the 'date' column in df_yearly
    merged_df = pd.merge(
        forecasts, inflation, left_on="year", right_on="year", how="inner"
    )

    # Drop the 'year' column if you don't need it in the final merged dataframe
    merged_df = merged_df.drop(columns=["year"])
    print(merged_df)

    mse = mean_squared_error(
        merged_df["predicted_mean"],
        forecasts["predicted_mean"].values
        + (forecasts["predicted_mean"].values * merged_df["rate"] / 100),
    )
    print(f"Mean Squared Error: {mse}")


data = read()
train, test = split(data)


model = train_model(train, 12, 2, 1)


def load_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


evaluate_model(model, test)
# model = load_model(2, 4)

# # Save the fitted model to a file using joblib
filename = "./data/2/4/smp_quotations_arima_model.joblib"
joblib.dump(model, filename)

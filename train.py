from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib


def read():
    smp_2_4 = pd.read_csv('./data/2/4/smp_quotations.csv')

    smp_2_4 = smp_2_4.sort_values(by='date')

    smp_2_4['date'] = pd.to_datetime(smp_2_4['date'])
    smp_2_4.set_index('date', inplace=True)

    return smp_2_4


def split(df):
    train_size = int(len(df) * 0.9)  # 80% for training, adjust as needed
    train, test = df[:train_size], df[train_size:]

    return train, test


def train_model(train_dataset, p, d, q):
    order = (p, d, q)
    model = ARIMA(train_dataset['price'], order=order)
    fitted_model = model.fit()

    return fitted_model


def evaluate_model(fitted_model, test_dataset):
    predictions = fitted_model.predict(start=len(train), end=len(train) + len(test) - 1)
    forecasts = fitted_model.forecast(len(test))
    mse = mean_squared_error(test_dataset['price'], predictions)
    forecasts_mse = mean_squared_error(test_dataset['price'], forecasts)
    print(f'Mean Squared Error: {mse}')
    print(f'Forecast Mean Squared Error: {forecasts_mse}')


data = read()
train, test = split(data)

model = train_model(train, 12, 2, 1)
evaluate_model(model, test)

# Save the fitted model to a file using joblib
filename = './data/2/4/smp_quotations_arima_model.joblib'
joblib.dump(model, filename)

# def read():
#     smp_2_4 = pd.read_csv('./data/2/4/smp_quotations.csv')
#     smp_2_4 = smp_2_4.sort_values(by='date')
#     smp_2_4['date'] = pd.to_datetime(smp_2_4['date'])
#     smp_2_4.set_index('date', inplace=True)
#     return smp_2_4
#
# def split(df):
#     train_size = int(len(df) * 0.9)
#     train, test = df[:train_size], df[train_size:]
#     return train, test
#
# def evaluate_model(fitted_model, test_dataset):
#     predictions = fitted_model.predict(n_periods=len(test_dataset))
#     mse = mean_squared_error(test_dataset['price'], predictions)
#     return mse
#
# data = read()
# train, test = split(data)
#
# # Auto ARIMA model selection
# model = auto_arima(train['price'], suppress_warnings=True, seasonal=False)
# print(f'Chosen ARIMA order: {model.order}')
#
# # Fit the model
# model.fit(train['price'])
#
# # Evaluate on the test set
# mse = evaluate_model(model, test)
# print(f'Mean Squared Error: {mse}')

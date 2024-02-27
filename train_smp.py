from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib


def read():
    smp_2_4 = pd.read_csv('./data/2/4/smp_quotations.csv')

    smp_2_4 = smp_2_4.sort_values(by='date')

    smp_2_4['date'] = pd.to_datetime(smp_2_4['date'])
    smp_2_4['date_b'] = smp_2_4['date']
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

    inflation = pd.read_csv('./data/2/inflation_rates.csv')
    inflation = inflation[inflation['data_interval'] == 'yearly']
    inflation = inflation.sort_values(by='date')

    inflation['date'] = pd.to_datetime(inflation['date'])
    inflation['date_a'] = inflation['date']
    inflation.set_index('date', inplace=True)



    # Extract the year from the date index in df_weekly
    test_dataset['year'] = test_dataset.index.year
    inflation['year'] = inflation.index.year

    # Merge based on the condition: year in df_weekly should match the 'date' column in df_yearly
    merged_df = pd.merge(test_dataset, inflation, left_on='year', right_on='year', how='inner')

    # Drop the 'year' column if you don't need it in the final merged dataframe
    merged_df = merged_df.drop(columns=['year'])


    mse = mean_squared_error(merged_df['price'], predictions.values + (predictions.values * merged_df['rate'] / 100))
    forecasts_mse = mean_squared_error(test_dataset['price'], forecasts)
    print(f'Mean Squared Error: {mse}')
    print(f'Forecast Mean Squared Error: {forecasts_mse}')


data = read()
train, test = split(data)


# model = train_model(train, 12, 2, 1)
def load_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


model = load_model(2, 4)
evaluate_model(model, test)

# # Save the fitted model to a file using joblib
# filename = './data/2/4/smp_quotations_arima_model.joblib'
# joblib.dump(model, filename)

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor



def load_smp_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


def load_production_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/production_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


def read_smp():
    smp_2_4 = pd.read_csv('./data/2/4/smp_quotations.csv')

    smp_2_4 = smp_2_4.sort_values(by='date')

    smp_2_4['date'] = pd.to_datetime(smp_2_4['date'])

    # smp_2_4.set_index('date', inplace=True)
    # smp_2_4['date'] = smp_2_4.index

    return smp_2_4


# a = read()

smp_model = load_smp_model(2, 4)
production_model = load_production_model(2, 4)

smp_data = read_smp()

smp_predictions = pd.DataFrame(smp_model.predict(start=0, end=1620))
smp_predictions['date'] = smp_predictions.index

merged_df = pd.merge(smp_predictions, smp_data, left_on=smp_predictions['date'], right_on=smp_data['date'], how='inner')

production_predictions = pd.DataFrame(production_model.predict(start=0, end=1620))

merged_df.set_index('date_x', inplace=True)

merged_df = merged_df[['predicted_mean', 'price']]

merged_df['year'] = merged_df.index.year
merged_df['month'] = merged_df.index.month

production_predictions['year'] = production_predictions.index.year
production_predictions['month'] = production_predictions.index.month

merged_df_2 = pd.merge(merged_df, production_predictions, left_on=['year', 'month'], right_on=['year', 'month'],
                     how='inner')

merged_df_2 = merged_df_2.rename(columns={"predicted_mean_x": "estimated_price", "predicted_mean_y": "production_value"})

print(merged_df_2)

def split(df):
    train_size = int(len(df) * 0.9)  # 80% for training, adjust as needed
    train, test = df[:train_size], df[train_size:]

    return train, test


train, test = split(merged_df_2)

def evaluate_model(fitted_model, test_dataset):
    predictions = fitted_model.predict(test_dataset[['estimated_price', 'production_value']])

    mse = mean_squared_error(test_dataset['price'], predictions)

    print(f'Mean Squared Error: {mse}')


regr = RandomForestRegressor(max_depth=4)
regr.fit(train[['estimated_price', 'production_value']], train['price'])


evaluate_model(regr, test)


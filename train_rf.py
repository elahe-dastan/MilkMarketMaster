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

    return smp_2_4


def prepare_data():
    smp_model = load_smp_model(2, 4)
    smp_predictions = pd.DataFrame(smp_model.predict(start=0, end=1620))
    smp_predictions['date'] = smp_predictions.index
    smp_predictions['year'] = smp_predictions.index.year
    smp_predictions['month'] = smp_predictions.index.month
    smp_predictions = smp_predictions.rename(columns={"predicted_mean": "estimated_price"})

    production_model = load_production_model(2, 4)
    production_predictions = pd.DataFrame(production_model.predict(start=0, end=1620))
    production_predictions['year'] = production_predictions.index.year
    production_predictions['month'] = production_predictions.index.month
    production_predictions = production_predictions.rename(columns={"predicted_mean": "production_value"})

    return smp_predictions, production_predictions


def feature_engineering(smp_predictions, production_predictions):
    smp_data = read_smp()
    smp = pd.merge(smp_predictions, smp_data, left_on=smp_predictions['date'], right_on=smp_data['date'], how='inner')
    smp.set_index('date_x', inplace=True)

    smp = smp[['estimated_price', 'price', 'year', 'month']]

    dataset = pd.merge(smp, production_predictions, left_on=['year', 'month'], right_on=['year', 'month'], how='inner')

    return dataset

def split(df):
    train_size = int(len(df) * 0.9)  # 80% for training, adjust as needed
    train, test = df[:train_size], df[train_size:]

    return train, test


def train_model(train):
    regr = RandomForestRegressor(max_depth=4)
    regr.fit(train[['estimated_price', 'production_value']], train['price'])

    return regr


def evaluate_model(fitted_model, test_dataset):
    predictions = fitted_model.predict(test_dataset[['estimated_price', 'production_value']])

    mse = mean_squared_error(test_dataset['price'], predictions)

    print(f'Mean Squared Error: {mse}')


smp_predictions, production_predictions = prepare_data()
dataset = feature_engineering(smp_predictions, production_predictions)
train, test = split(dataset)
model = train_model(train)
evaluate_model(model, test)

filename = "./data/2/4/rf_model.joblib"
joblib.dump(model, filename)

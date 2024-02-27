# main.py

import joblib
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import logging

app = FastAPI()


def load_model(country_id, product_id):
    path = f"./data/{country_id}/{product_id}/smp_quotations_arima_model.joblib"
    # Load the ARIMA model
    # HANDLE EXCEPTION **********************
    model = joblib.load(path)

    return model


def load_inflation(country_id):
    path = f"./data/{country_id}/inflation_rates.csv"
    # Load the inflation csv
    # HANDLE EXCEPTION **********************
    inflation = pd.read_csv(path)
    # FOR NOW --> Quarterly
    inflation = inflation[inflation['data_interval'] == 'yearly']
    inflation = inflation.sort_values(by='date')
    inflation = inflation[['date', 'rate']]

    inflation['date'] = pd.to_datetime(inflation['date'])
    inflation.set_index('date', inplace=True)

    return inflation

def merge_forecast_inflation(forecast_df, inflation):
    # Extract the year from the date index in df_weekly
    forecast_df['year'] = forecast_df.index.year

    inflation['year'] = inflation.index.year

    # Merge based on the condition: year in df_weekly should match the 'date' column in df_yearly
    forecast_inflation_merged = pd.merge(forecast_df, inflation, left_on='year', right_on='year', how='inner')

    return forecast_inflation_merged


def maximize_profit(forecast_df):
    # Find the index of the maximum value
    max_index = forecast_df['adjusted_price'].idxmax()

    # Find the index of the minimum value before the maximum value
    min_index_before_max = forecast_df['adjusted_price'].loc[:max_index].idxmin()

    # Calculate the maximum profit
    max_profit = forecast_df['adjusted_price'][max_index] - forecast_df['adjusted_price'][min_index_before_max]

    # Print the result
    # print(f"Buy at index {min_index_before_max}, sell at index {max_index}, max profit: {max_profit}")
    return f"Buy at index {forecast_df['forecast_date'][min_index_before_max]}, sell at index {forecast_df['forecast_date'][max_index]}, max profit: {max_profit}"


@app.get("/predict")
def predict(country_id: int = Query(..., title="Country ID", description="ID of the country"),
            product_id: int = Query(..., title="Product ID", description="ID of the product")):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        logging.log(1, 'hi')
        # Make prediction using the loaded ARIMA model
        # steps = 16 means 4 months
        model = load_model(country_id, product_id)
        forecasts = model.forecast(steps=16)  # Adjust as needed

        last_date = pd.to_datetime('2023-12-21')

        forecast_df = pd.DataFrame({
            'forecast_date': pd.date_range(start=last_date + pd.DateOffset(1), periods=16, freq='W-THU'),
            'forecast_index': pd.date_range(start=last_date + pd.DateOffset(1), periods=16, freq='W-THU'),
            'forecast_price': forecasts
        })
        forecast_df.set_index('forecast_index', inplace=True)


        inflation = load_inflation(country_id)
        forecast_inflation_merged = merge_forecast_inflation(forecast_df, inflation)

        forecast_inflation_merged['adjusted_price'] = forecast_inflation_merged['forecast_price'] + (forecast_inflation_merged['forecast_price'] * forecast_inflation_merged['rate'] / 100)

        max_profit = maximize_profit(forecast_inflation_merged)
        # return {"prediction": prediction.iloc[0]}
        return {"max_profit: ": max_profit}
    except Exception as e:
        logging.info('a')
        logging.info(e)
        logging.info('b')
        raise HTTPException(status_code=500, detail=str(e))

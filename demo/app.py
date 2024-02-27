# streamlit_app/app.py

import pandas as pd
import requests
import streamlit as st

endpoint = "http://127.0.0.1:8080/predict"


def main():
    st.title("Milk Market Master 🥛")

    try:
        value = int(st.number_input("Number of steps", value=16, step=1))
    except ValueError:
        st.error("steps should be a number")
        st.stop()

    param = st.selectbox("Parameter", options=["price", "production"])
    if param is None:
        st.error("parameter must be selected")
        st.stop()

    # Make a request to the FastAPI API
    if st.button("Predict"):
        response = make_prediction(value, param)
        match param:
            case "price":
                df: pd.DataFrame = pd.DataFrame.from_dict(response)
                df = df.set_index(pd.to_datetime(df["forecast_date"]).dt.date)
                st.line_chart(df["adjusted_price"])
            case "production":
                df: pd.DataFrame = pd.DataFrame.from_dict(response)
                df = df.set_index(pd.to_datetime(df["date"]).dt.date)
                st.line_chart(df["value"])


def make_prediction(steps: int, param: str):
    response = requests.get(
        endpoint,
        params={
            "product_id": 4,
            "country_id": 2,
            "steps": steps,
            "df": True,
            "param": param,
        },
    )
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    main()

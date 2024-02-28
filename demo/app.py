# streamlit_app/app.py

import pandas as pd
import requests
import streamlit as st

endpoint = "http://127.0.0.1:8080/predict"

COUNTRIES = {
    2: "United States",
    67: "Germany",
    129: "Netherlands",
    131: "New Zealand",
    1: "Europe",
}

PRODUCTS = {4: "SMP (Food)", 1: "Milk"}


def main():
    st.title("Milk Market Master 🥛")

    country_id = st.selectbox(
        "Country",
        options=COUNTRIES.keys(),
        format_func=lambda k: COUNTRIES.get(k),
    )
    if country_id is None:
        st.error("country id must be selected")
        st.stop()

    product_id = st.selectbox(
        "Product",
        options=PRODUCTS.keys(),
        format_func=lambda k: PRODUCTS.get(k),
    )
    if product_id is None:
        st.error("product id must be selected")
        st.stop()

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
        response = make_prediction(value, param, product_id, country_id)
        match param:
            case "price":
                df: pd.DataFrame = pd.DataFrame.from_dict(response)
                df.index = pd.to_datetime(df.index)
                st.line_chart(df["adjusted_price"])
            case "production":
                sr: pd.Series = pd.Series(response)
                sr.index = pd.to_datetime(sr.index)
                st.line_chart(sr)


def make_prediction(steps: int, param: str, product_id: int, country_id: int):
    response = requests.get(
        endpoint,
        params={
            "product_id": product_id,
            "country_id": country_id,
            "steps": steps,
            "df": True,
            "param": param,
        },
    )
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    main()

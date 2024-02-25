# streamlit_app/app.py

import streamlit as st
import requests

# Define the FastAPI API endpoint
api_endpoint = "http://fastapi-app:80/predict"  # Use the Docker service name


def main():
    st.title("Streamlit App")

    # Interactive input form
    value = st.number_input("Enter a country id:", value=2, step=0.1)

    # Make a request to the FastAPI API
    if st.button("Predict"):
        response = make_prediction(value)
        st.success(f"Prediction: {response['prediction']}")


def make_prediction(value):
    # payload = {"value": value}
    response = requests.get(api_endpoint)
    return response.json()


if __name__ == "__main__":
    main()

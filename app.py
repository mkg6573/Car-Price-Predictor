import streamlit as st
import pandas as pd
import joblib
import numpy as np

LRmodel = joblib.load("LineraRegressionModel.pkl")

data = pd.read_csv(r"clean_car.csv")
company = data['company'].unique()
name = data['name'].unique()

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")

# Input fields
company = st.selectbox("Select Car Company", company)
if(company):
    model_name = []
    for i in name:
        if company in i:
            model_name.append(i)
    model = st.selectbox("Enter Car Model",model_name)

fuel_type = st.selectbox("Select Fuel Type", ["Petrol", "Diesel"])
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)

# Predict button
if st.button("Predict Price"):
    # --- Prediction logic will go here ---
    prediction = LRmodel.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                            data=np.array([model,company,year,kms_driven,fuel_type]).reshape(1,5)))[0]
    st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")
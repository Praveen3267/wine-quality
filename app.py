import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature names in correct order
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']

st.title("🍷 Wine Quality Prediction App")

st.markdown("Enter the wine's physicochemical properties:")

# Create input fields
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature}", format="%.4f")
    input_data.append(val)

if st.button("Predict Quality"):
    try:
        # Create DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(f"Predicted Wine Quality: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error: {str(e)}")

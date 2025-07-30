import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Set up Streamlit app
st.title("USO Price Prediction App")
st.write("Enter the values below to predict the USO Closing Price")

# Input fields for the user — adjust these to match your top features
open_val = st.number_input("Open", value=79.12)
high_val = st.number_input("High", value=79.48)
low_val = st.number_input("Low", value=78.93)
volume_val = st.number_input("Volume", value=15960100)
year = st.number_input("Year", value=2015)
month = st.number_input("Month", value=1)

# Add more features if your model needs them
# Example:
# gdx_open = st.number_input("GDX_Open", value=21.5)

# Collect input
input_data = np.array([[open_val, high_val, low_val, volume_val, year, month]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted USO Close Price: {prediction[0]:.2f}")

import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set up the title of the app
st.title("Energy Consumption Prediction for Smart Homes")

# Set up the input fields for user data
st.header("Enter the following information:")

T1 = st.number_input("Temperature in Kitchen (T1) (°C)", min_value=-50.0, max_value=50.0, value=22.0)
RH_1 = st.number_input("Humidity in Kitchen (RH_1) (%)", min_value=0.0, max_value=100.0, value=45.0)
T_out = st.number_input("Outdoor Temperature (T_out) (°C)", min_value=-50.0, max_value=50.0, value=15.0)

# Create a button for prediction
if st.button("Predict Energy Consumption"):
    # Prepare the input data for prediction
    input_data = np.array([[T1, RH_1, T_out]])

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the model
    prediction = model.predict(input_data_scaled)

    # Display the predicted energy consumption
    st.write(f"Predicted Energy Consumption: {prediction[0]} Wh")
    st.write("This prediction is based on the provided temperature and humidity values.")

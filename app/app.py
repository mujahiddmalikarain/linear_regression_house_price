import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")

# Streamlit App UI
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction")
st.write("This app predicts house price using area, bedrooms, and bathrooms.")

# User inputs
area = st.number_input("📏 Area (sq ft)", min_value=100, max_value=10000, value=1500)
bedrooms = st.slider("🛏️ Bedrooms", 1, 10, 3)
bathrooms = st.slider("🛁 Bathrooms", 1, 10, 2)

# Predict Button
if st.button("🔮 Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")

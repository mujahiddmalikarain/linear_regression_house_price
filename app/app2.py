import streamlit as st
import joblib
import numpy as np

# Load both models
lr_model = joblib.load("models/model_linear.pkl")
rf_model = joblib.load("models/model_rf.pkl")

# Streamlit UI
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 House Price Predictor")
st.write("Choose a model and enter details to predict house price.")

# Model Selection
model_choice = st.selectbox("📊 Select Prediction Model", ["Linear Regression", "Random Forest"])

# User Inputs
area = st.number_input("📏 Area (sq ft)", min_value=100, max_value=10000, value=1500)
bedrooms = st.slider("🛏️ Bedrooms", 1, 10, 3)
bathrooms = st.slider("🛁 Bathrooms", 1, 10, 2)

# Predict Button
if st.button("🔮 Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)
    else:
        prediction = rf_model.predict(input_data)

    st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")

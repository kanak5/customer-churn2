import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('churn_dl_model.h5')
scaler = joblib.load('churn_scaler.pkl')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("🔍 Customer Churn Prediction App")
st.markdown("Enter customer details to predict if they will churn.")

# User inputs
credit_score = st.slider("Credit Score", 300, 850, 650)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (years with bank)", 0, 10, 3)
balance = st.number_input("Balance ($)", 0.0, 250000.0, 60000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary ($)", 10000.0, 250000.0, 50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Convert inputs to model-ready format
gender = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# Final input vector
input_data = np.array([[credit_score, gender, age, tenure, balance, num_of_products,
                        has_cr_card, is_active, salary, geo_germany, geo_spain]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    pred = model.predict(input_scaled)
    prob = float(pred[0][0])
    st.markdown(f"**Churn Probability:** `{prob:.2%}`")
    if prob > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")

st.markdown("---")
st.caption("AI Project by Kanak Sethi | Churn Prediction using Deep Learning")

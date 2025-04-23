import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Churn Predictor", layout="centered") 

# === LOAD MODEL & SCALER ===
model = load_model('churn_dl_model.h5')
scaler = joblib.load('churn_scaler.pkl')

# === HELPER: LOAD ANIMATION ===
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_churn = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ktwnwv5m.json")

# === SET BACKGROUND IMAGE ===
CUSTOM_IMAGE_URL = 'https://imgur.com/a/51bSVIB'  # Replace with your actual uploaded image URL

st.markdown("""
    <style>
    .stApp > div:first-child {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(128, 0, 255, 0.3), 
                    0 4px 8px rgba(0, 128, 255, 0.2);
        transform: perspective(1000px) translateZ(10px);
        transition: all 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# === STREAMLIT APP ===


# === HEADER ===
st.title("\U0001F52E Customer Churn Prediction")

col1, col2 = st.columns([2, 3])
with col1:
    st_lottie(lottie_churn, height=200)
with col2:
    st.subheader("Know if your customer will stay or leave!")
    st.caption("Powered by a Deep Learning AI Model")

# === INPUT FORM ===
with st.form("churn_form"):
    st.markdown("### \U0001F9FE Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("\U0001F4B3 Credit Score", 300, 850, 650)
        gender = st.selectbox("\U0001F9CD Gender", ["Male", "Female"])
        age = st.slider("\U0001F382 Age", 18, 100, 40)
        tenure = st.slider("\U0001F4C6 Tenure (years)", 0, 10, 3)
        balance = st.number_input("\U0001F4B6 Balance ($)", 0.0, 250000.0, 60000.0)

    with col2:
        num_of_products = st.selectbox("\U0001F6CD️ Number of Products", [1, 2, 3, 4])
        has_cr_card = st.radio("\U0001F4B3 Has Credit Card?", ["Yes", "No"])
        is_active = st.radio("\U0001F525 Is Active Member?", ["Yes", "No"])
        salary = st.number_input("\U0001F4B0 Estimated Salary ($)", 10000.0, 250000.0, 50000.0)
        geography = st.selectbox("\U0001F30D Geography", ["France", "Germany", "Spain"])

    submitted = st.form_submit_button("\U0001F50D Predict Churn")

# === PREDICTION ===
if submitted:
    gender = 1 if gender == "Male" else 0
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active = 1 if is_active == "Yes" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    input_data = np.array([[credit_score, gender, age, tenure, balance, num_of_products,
                            has_cr_card, is_active, salary, geo_germany, geo_spain]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)
    prob = float(pred[0][0])

    st.markdown("---")
    st.subheader("\U0001F4CA Prediction Result")
    st.markdown(f"**Churn Probability:** `{prob:.2%}`")
    st.progress(min(prob, 1.0))

    if prob > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")

    st.caption("🎯 A threshold of `30%` was used to determine churn likelihood.")

# === FOOTER ===
st.markdown("---")
st.caption("📘 AI Project  — Churn Prediction with Deep Learning")

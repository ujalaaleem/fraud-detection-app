import streamlit as st, joblib, pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detector")
amount = st.number_input("Amount", 0.0, value=50.0)
hour = st.slider("Hour of day", 0, 23, 12)
distance = st.number_input("Distance from home (km)", 0.0, value=5.0)
txns = st.number_input("Transactions in last 24h", 0, value=1)
risk = st.slider("Merchant risk score", 0.0, 1.0, 0.2)

if st.button("Predict"):
    row = pd.DataFrame([[amount, hour, distance, txns, risk]],
                       columns=["amount", "hour_of_day", "distance_from_home_km",
                                "transactions_last_24h", "merchant_risk_score"])
    p = model.predict_proba(scaler.transform(row))[0, 1]
    st.metric("Fraud probability", f"{p:.1%}")
    if p >= 0.5:
        st.error("Likely FRAUD")
    else:
        st.success("Looks legitimate")

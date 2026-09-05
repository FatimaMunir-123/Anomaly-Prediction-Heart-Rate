
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Rate Anomaly Detection",
    page_icon="❤️",
    layout="wide"
)

pipeline = joblib.load("heart_rate_anomaly_model.pkl")

features = [
    "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU",
    "HF", "HF_PCT", "HF_NU", "TP", "LF_HF", "HF_LF"
]

st.title("❤️ Heart Rate Anomaly Detection")
st.write("Machine Learning System for Frequency-Domain Heart Rate Anomaly Detection")

st.subheader("Enter Heart Rate Features")

values = {}

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(features):
    if i % 3 == 0:
        container = col1
    elif i % 3 == 1:
        container = col2
    else:
        container = col3

    values[feature] = container.number_input(
        feature,
        value=0.0,
        format="%.6f"
    )

if st.button("🔍 Predict", use_container_width=True):

    input_data = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features
    )

    prediction = pipeline.predict(input_data)[0]
    score = pipeline.decision_function(input_data)[0]

    if prediction == 1:
        st.success("🟢 NORMAL")
        st.write("The input pattern is statistically normal.")
    else:
        st.error("🔴 ANOMALY")
        st.write("The input pattern is statistically unusual.")

    st.metric("Anomaly Score", f"{score:.4f}")

st.divider()

st.subheader("🤖 Model Information")

c1, c2, c3 = st.columns(3)

c1.metric("Algorithm", "Isolation Forest")
c2.metric("Trees", "200")
c3.metric("Contamination", "5%")

st.info(
    "This system detects statistically unusual patterns in heart-rate "
    "frequency-domain features. It is not a medical diagnosis."
)

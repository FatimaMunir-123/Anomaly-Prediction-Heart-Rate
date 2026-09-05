import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Rate Anomaly Detection",
    page_icon="❤️",
    layout="wide"
)

# Load trained machine learning model
pipeline = joblib.load("heart_rate_anomaly_model.pkl")

# Model features
features = [
    "VLF",
    "VLF_PCT",
    "LF",
    "LF_PCT",
    "LF_NU",
    "HF",
    "HF_PCT",
    "HF_NU",
    "TP",
    "LF_HF",
    "HF_LF"
]

# Title
st.title("❤️ Heart Rate Anomaly Detection")
st.write(
    "Machine Learning System for Frequency-Domain "
    "Heart Rate Anomaly Detection"
)

st.divider()

# Input section
st.subheader("🫀 Enter Heart Rate Features")
st.write(
    "Enter the frequency-domain heart rate values below "
    "and click Predict to detect whether the pattern is "
    "normal or anomalous."
)

values = {}

# Three-column layout
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

st.divider()

# Prediction button
if st.button("🔍 Predict", use_container_width=True):
    # Create input dataframe
    input_data = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )

    # Make prediction
    prediction = pipeline.predict(input_data)[0]

    # Get anomaly score
    score = pipeline.decision_function(input_data)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("🟢 NORMAL")
        st.write(
            "The input pattern is statistically normal "
            "according to the trained Isolation Forest model."
        )
    else:
        st.error("🔴 ANOMALY")
        st.write(
            "The input pattern is statistically unusual "
            "according to the trained Isolation Forest model."
        )

    st.metric(
        "Anomaly Score",
        f"{score:.4f}"
    )

    # Show entered values
    with st.expander("View Input Values"):
        st.dataframe(input_data, use_container_width=True)

st.divider()

# Model information
st.subheader("🤖 Model Information")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Algorithm",
    "Isolation Forest"
)

c2.metric(
    "Trees",
    "200"
)

c3.metric(
    "Contamination",
    "5%"
)

st.divider()

# Disclaimer
st.info(
    "ℹ️ This system detects statistically unusual patterns "
    "in heart-rate frequency-domain features. "
    "It is not a medical diagnosis."
)

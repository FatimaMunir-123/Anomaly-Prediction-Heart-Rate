import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Rate Anomaly Detection",
    page_icon="❤️",
    layout="wide"
)

# Load trained machine learning model
pipeline = joblib.load("heart_rate_anomaly_model (1).pkl")

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
st.subheader("🫀 Heart Rate Monitoring")

st.write(
    "Adjust the heart-rate frequency features using the sliders "
    "and monitor the pattern before running the prediction."
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

    if feature in ["VLF", "LF", "HF", "TP"]:
        values[feature] = container.slider(
            feature,
            min_value=0.0,
            max_value=50000.0,
            value=1000.0,
            step=0.01
        )

    elif feature in [
        "VLF_PCT",
        "LF_PCT",
        "HF_PCT",
        "LF_NU",
        "HF_NU"
    ]:
        values[feature] = container.slider(
            feature,
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1
        )

    else:
        values[feature] = container.slider(
            feature,
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.01
        )

st.divider()

# Live chart
st.subheader("📈 Live Feature Monitor")

chart_data = pd.DataFrame({
    "Feature": features,
    "Value": [values[feature] for feature in features]
})

st.line_chart(
    chart_data.set_index("Feature"),
    use_container_width=True
)

st.divider()

# Buttons
predict_col, monitor_col = st.columns(2)

with predict_col:
    predict_button = st.button(
        "🔍 Predict Anomaly",
        use_container_width=True
    )

with monitor_col:
    monitor_button = st.button(
        "🖥️ Monitor",
        use_container_width=True
    )

# Monitor
if monitor_button:

    st.subheader("🖥️ Monitoring Status")

    st.success(
        "🟢 Monitoring Active"
    )

    st.write(
        "The current heart-rate frequency-domain features "
        "are being monitored."
    )

    monitor_data = pd.DataFrame({
        "Feature": features,
        "Current Value": [
            values[feature] for feature in features
        ]
    })

    st.dataframe(
        monitor_data,
        use_container_width=True
    )

# Prediction
if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )

    # Make prediction
    prediction = pipeline.predict(input_data)[0]

    # Get anomaly score
    score = pipeline.decision_function(input_data)[0]

    st.divider()

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
    with st.expander("📋 View Input Values"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

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

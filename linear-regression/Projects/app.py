
from pathlib import Path

import numpy as np
import streamlit as st

from src.model import predict

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.write(
    "Predict house prices using our Linear Regression model "
    "built from scratch with NumPy."
)

# Load saved model


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "housing_linear_regression.npz"
)


@st.cache_resource
def load_model():

    with np.load(MODEL_PATH) as model:

        weights = model["weights"].copy()
        bias = float(model["bias"])

        feature_mean = model["feature_mean"].copy()
        feature_std = model["feature_std"].copy()

        target_mean = float(model["target_mean"])
        target_std = float(model["target_std"])

        feature_names = model["feature_names"].tolist()

    return (
        weights,
        bias,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        feature_names
    )


if not MODEL_PATH.exists():

    st.error(
        "Trained model not found. Please run train.py first."
    )
    st.stop()


try:

    (
        weights,
        bias,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        feature_names
    ) = load_model()

except Exception as error:

    st.error(f"Could not load model: {error}")
    st.stop()


# User input


st.subheader("Enter House Details")
with st.form("house_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=1,
            max_value=50000,
            value=5000,
            step=100
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

    with col2:

        stories = st.number_input(
            "Stories",
            min_value=1,
            max_value=10,
            value=2
        )

        parking = st.number_input(
            "Parking spaces",
            min_value=0,
            max_value=10,
            value=1
        )

    submitted = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )


# Prediction

if submitted:

    input_values = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking
    }

    try:

        # Arrange input in the original training feature order
        input_array = np.array(
            [[input_values[name] for name in feature_names]],
            dtype=float
        )

        # Apply the same feature scaling used during training
        input_scaled = (
            input_array - feature_mean
        ) / feature_std

        # Predict scaled house price
        prediction_scaled = predict(
            input_scaled,
            weights,
            bias
        )

        # Convert prediction back to actual price
        predicted_price = float(
            prediction_scaled.item() * target_std
            + target_mean
        )

        if not np.isfinite(predicted_price):
            raise ValueError("Prediction is not a valid number.")

        # Display result
        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            label="Estimated House Price",
            value=f"₹{predicted_price:,.0f}"
        )

        st.success("Prediction generated successfully!")

    except (KeyError, ValueError, TypeError) as error:

        st.error(f"Prediction failed: {error}")


st.caption(
    "Built using Python, NumPy and Streamlit. "
    "This educational model uses five numerical features; "
    "actual house prices may depend on additional factors."
)

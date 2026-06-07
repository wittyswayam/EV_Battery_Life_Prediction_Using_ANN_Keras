"""Reusable Streamlit UI components.

Each function here renders a specific section of the interface and returns
nothing — they write directly to the Streamlit surface. Keeping them here
means the page modules stay thin and easy to reason about.
"""

import streamlit as st

from app.config import VALID_CYCLE_TYPES


def render_header() -> None:
    st.markdown(
        "<h1 style='text-align:center'>&#9889; EV Battery Life Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray'>"
        "Predict ambient operating temperature from electrochemical cycle measurements "
        "using an ANN trained on the NASA PCoE Battery Dataset."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()


def render_input_form() -> tuple[str, float, float, float]:
    """Render the four input fields and return their current values."""
    col1, col2 = st.columns(2)

    with col1:
        cycle_type = st.selectbox(
            label="Cycle Type",
            options=list(VALID_CYCLE_TYPES),
            help="Type of battery test cycle recorded in the NASA dataset.",
        )
        capacity = st.number_input(
            label="Capacity (Ah)",
            min_value=0.0,
            value=1.674,
            step=0.001,
            format="%.4f",
            help="Measured battery capacity in ampere-hours. Only populated for discharge cycles.",
        )

    with col2:
        re = st.number_input(
            label="Re (Ω) — Electrolyte Resistance",
            value=0.056058,
            step=0.000001,
            format="%.6f",
            help="Electrolyte resistance in ohms from impedance spectroscopy.",
        )
        rct = st.number_input(
            label="Rct (Ω) — Charge-Transfer Resistance",
            value=0.200970,
            step=0.000001,
            format="%.6f",
            help="Charge-transfer resistance in ohms from impedance spectroscopy.",
        )

    return cycle_type, capacity, re, rct


def render_prediction_result(temperature: float) -> None:
    """Display the prediction result with context."""
    st.success(f"Predicted Ambient Temperature: **{temperature:.2f} °C**")
    st.caption(
        "The model predicts the ambient temperature at which this battery cycle occurred. "
        "Training data covers three discrete temperatures: 4 °C, 24 °C, and 44 °C. "
        "Outputs outside this range may indicate out-of-distribution inputs."
    )


def render_validation_error(message: str) -> None:
    st.error(f"❌ Validation Error: {message}")


def render_runtime_error(message: str) -> None:
    st.error(f"⚠️ Prediction failed: {message}")
    st.info(
        "If this error mentions a missing artifact, ensure you have run the training "
        "notebook to generate `artifacts/battery_life_model.pkl`, `scaler.pkl`, "
        "and `label_encoder.pkl`."
    )


def render_sidebar_info() -> None:
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **Dataset:** NASA PCoE Battery Dataset  
            7,565 rows covering 34 Li-ion cells (B0025–B0056)  
            Temperatures: 4 °C, 24 °C, 44 °C

            **Model:** Feed-forward ANN  
            Architecture: 4 → 64 (ReLU) → 32 (ReLU) → 1 (linear)  
            Optimizer: Adam | Loss: MSE | Epochs: 150

            **Preprocessing:**  
            - LabelEncoder on `type`  
            - MinMaxScaler on all features  
            - Mean imputation for structural NaNs
            """
        )
        st.divider()
        st.markdown(
            "[GitHub](https://github.com/wittyswayam/EV_Battery_Life_Prediction_Using_ANN_Keras) "
            "| [NASA Dataset](https://www.nasa.gov/intelligent-systems-division/"
            "discovery-and-systems-health/pcoe/pcoe-data-set-repository/)"
        )

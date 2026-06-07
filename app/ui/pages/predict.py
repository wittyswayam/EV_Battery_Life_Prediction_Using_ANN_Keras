"""Prediction page — the main application view."""

import streamlit as st

from app.predictor import PredictionService
from app.ui.components import (
    render_header,
    render_input_form,
    render_prediction_result,
    render_runtime_error,
    render_sidebar_info,
    render_validation_error,
)
from app.validators import ValidationError


@st.cache_resource(show_spinner="Loading model artifacts...")
def _load_service() -> PredictionService:
    """
    Load once per server session, cached by Streamlit across reruns.
    st.cache_resource is appropriate here because PredictionService holds
    large objects (model weights, scaler state) that are expensive to reload
    and are safe to share across users in a single-process deployment.
    """
    return PredictionService()


def render() -> None:
    render_sidebar_info()
    render_header()

    try:
        service = _load_service()
    except Exception as exc:
        render_runtime_error(str(exc))
        st.stop()
        return

    cycle_type, capacity, re, rct = render_input_form()

    st.markdown("")
    if st.button("Predict Ambient Temperature", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            try:
                result = service.predict(cycle_type, capacity, re, rct)
                render_prediction_result(result)
            except ValidationError as exc:
                render_validation_error(str(exc))
            except Exception as exc:
                render_runtime_error(str(exc))

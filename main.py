"""Application entry point.

Run with:
    streamlit run main.py
"""

from app.config import APP_ICON, APP_TITLE, PAGE_LAYOUT
from app.ui.pages import predict
import streamlit as st

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

predict.render()

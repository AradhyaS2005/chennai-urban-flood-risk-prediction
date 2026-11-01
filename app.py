
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Chennai Flood Risk AI",
    page_icon="rainfall",
    layout="wide"
)

# Dashboard heading
st.title("Chennai Rainfall & Flood Risk Dashboard")

st.markdown(
    """
    An AI-powered prototype for next-day rainfall prediction
    using historical weather data from Chennai.
    """
)

st.divider()

# Model evaluation metrics from our Colab results
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Gradient Boosting MAE", "4.423 mm")

with col2:
    st.metric("Random Forest RMSE", "12.030 mm")

with col3:
    st.metric("Random Forest R²", "0.215")

st.info(
    "These metrics were measured on the held-out test period. "
    "They describe rainfall prediction performance, not flood prediction accuracy."
)

st.divider()

# Rainfall chart placeholder
st.subheader("Historical Rainfall")

st.write(
    "Our next step is to load the historical weather dataset "
    "and display daily and monthly rainfall trends here."
)

st.success("Dashboard foundation loaded successfully!")

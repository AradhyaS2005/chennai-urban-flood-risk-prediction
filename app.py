
import streamlit as st
import pandas as pd
import joblib
from datetime import date
import requests

# Page configuration
st.set_page_config(
    page_title="Chennai Rainfall Forecast",
    page_icon="🌧️",
    layout="wide"
)

# Cache the historical weather data for 1 hour to avoid repeated API calls
@st.cache_data(ttl=3600)
def fetch_historical_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "start_date": "2010-01-01",
        "end_date": "2025-12-31",
        "daily": [
            "precipitation_sum",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min"
        ],
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()["daily"]

    weather = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "rainfall_mm": data["precipitation_sum"],
        "temperature_mean_c": data["temperature_2m_mean"],
        "temperature_max_c": data["temperature_2m_max"],
        "temperature_min_c": data["temperature_2m_min"]
    })

    return weather


# Load trained models once
@st.cache_resource
def load_models():
    lr = joblib.load("linear_regression.pkl")
    rf = joblib.load("random_forest.pkl")
    gb = joblib.load("gradient_boosting.pkl")
    features = joblib.load("feature_names.pkl")
    return lr, rf, gb, features


try:
    lr_model, rf_model, gb_model, features = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()


# Dashboard heading
st.title("Chennai Rainfall Forecast Dashboard")
st.write("Predict next-day rainfall using your trained machine learning models.")

st.info(
    "This prototype predicts rainfall, not flood probability. "
    "Enter weather observations and recent rainfall history below."
)

st.divider()

# Evaluation results from our held-out test data in Colab
st.header("Model Performance Comparison")
performance = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE (mm)": [4.644, 4.579, 4.423],
    "RMSE (mm)": [12.203, 12.030, 12.311],
    "R² Score": [0.192, 0.215, 0.177]
})

st.dataframe(
    performance,
    hide_index=True,
    use_container_width=True
)

st.subheader("MAE Comparison")
st.bar_chart(performance.set_index("Model")[["MAE (mm)"]])

st.subheader("RMSE Comparison")
st.bar_chart(performance.set_index("Model")[["RMSE (mm)"]])

st.subheader("R² Score Comparison")
st.bar_chart(performance.set_index("Model")[["R² Score"]])

st.caption(
    "Metrics are from the held-out test period in Google Colab. "
    "Lower MAE and RMSE are better; higher R² is better."
)



# Prediction inputs
header_col, refresh_col = st.columns([4, 1])

with header_col:
    st.header("Predict Tomorrow's Rainfall")

with refresh_col:
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

with st.form("rainfall_prediction_form"):
    st.subheader("Today's weather")

    col1, col2, col3 = st.columns(3)

    with col1:
        rainfall = st.number_input(
            "Today's rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0
        )

    with col2:
        temp_mean = st.number_input(
            "Mean temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.0
        )

    with col3:
        temp_max = st.number_input(
            "Maximum temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=32.0
        )

    temp_min = st.number_input(
        "Minimum temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0
    )

    st.subheader("Recent rainfall history")

    col4, col5, col6 = st.columns(3)

    with col4:
        previous_day = st.number_input(
            "Rainfall 1 day ago (mm)",
            min_value=0.0,
            value=0.0
        )

    with col5:
        three_days_ago = st.number_input(
            "Rainfall 3 days ago (mm)",
            min_value=0.0,
            value=0.0
        )

    with col6:
        seven_days_ago = st.number_input(
            "Rainfall 7 days ago (mm)",
            min_value=0.0,
            value=0.0
        )

    rainfall_avg = st.number_input(
        "Average rainfall over previous 7 days (mm/day)",
        min_value=0.0,
        value=0.0
    )

    today = st.date_input(
        "Date of today's observations",
        value=date.today()
    )

    predict_button = st.form_submit_button(
        "Predict Tomorrow's Rainfall"
    )


if predict_button:
    input_values = {
        "rainfall_mm": rainfall,
        "temperature_mean_c": temp_mean,
        "temperature_max_c": temp_max,
        "temperature_min_c": temp_min,
        "month": today.month,
        "day_of_year": today.timetuple().tm_yday,
        "rainfall_previous_day": previous_day,
        "rainfall_3_days_ago": three_days_ago,
        "rainfall_7_days_ago": seven_days_ago,
        "rainfall_7_day_avg": rainfall_avg
    }

    input_df = pd.DataFrame(
        [[input_values[name] for name in features]],
        columns=features
    )

    predictions = {
        "Linear Regression": float(lr_model.predict(input_df)[0]),
        "Random Forest": float(rf_model.predict(input_df)[0]),
        "Gradient Boosting": float(gb_model.predict(input_df)[0])
    }

    st.subheader("Predicted rainfall for tomorrow")

    cols = st.columns(3)

    for col, (model_name, prediction) in zip(cols, predictions.items()):
        with col:
            st.metric(model_name, f"{max(0, prediction):.2f} mm")

    st.caption(
        "These are model estimates, not guaranteed rainfall amounts. "
        "Predictions depend on the quality of the input data and model performance."
    )

# charts (daily and monthly rainfall trends)

st.divider()

st.header("Historical Rainfall Analysis")

try:
    weather = fetch_historical_weather()

    st.success(f"Loaded {len(weather):,} daily weather records.")

    # Daily rainfall trend
    st.subheader("Daily Rainfall Trend")

    daily_data = weather.set_index("date")[["rainfall_mm"]]
    st.line_chart(daily_data)

    # Average daily rainfall for each calendar month
    st.subheader("Average Daily Rainfall by Month")

    monthly_data = weather.copy()
    monthly_data["month"] = monthly_data["date"].dt.month

    monthly_avg = (
        monthly_data.groupby("month")["rainfall_mm"]
        .mean()
        .reindex(range(1, 13))
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    monthly_chart = pd.DataFrame({
        "Month": month_names,
        "Average Rainfall (mm/day)": monthly_avg.values
    }).set_index("Month")

    st.bar_chart(monthly_chart)

    st.caption(
        "Monthly values represent average daily rainfall across all "
        "available years, not the total rainfall for each month."
    )

except Exception as e:
    st.error(f"Could not display historical rainfall charts: {e}")



st.divider()




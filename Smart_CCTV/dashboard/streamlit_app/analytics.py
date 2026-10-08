import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime

# --- Streamlit page setup ---
st.set_page_config(page_title="Analytics", layout="wide")
st.title("📈 Smart CCTV - Alert Analytics")

# --- Fetch alert data from Flask API ---
import os

API_TOKEN = os.getenv("SMART_CCTV_ADMIN_TOKEN", "")

try:
    response = requests.get(
        "http://localhost:5000/alerts",
        headers={"Authorization": API_TOKEN},
    )
    response.raise_for_status()
    alerts = response.json()
except Exception as e:
    st.error(f"⚠️ Failed to fetch alerts: {e}")
    alerts = []

# --- Convert to DataFrame ---
if isinstance(alerts, dict):
    df = pd.DataFrame([alerts])
else:
    df = pd.DataFrame(alerts)

# --- Check if DataFrame has content ---
if not df.empty:
    # ✅ Ensure timestamp exists (add current time if missing)
    if 'timestamp' not in df.columns:
        df['timestamp'] = [datetime.now() for _ in range(len(df))]

    # ✅ Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with invalid timestamps (optional safeguard)
    df = df.dropna(subset=['timestamp'])

    # Add hour column for chart grouping
    df['hour'] = df['timestamp'].dt.hour

    st.subheader("📊 Alerts per Hour")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('hour:O', title='Hour of Day'),
            y=alt.Y('count()', title='Number of Alerts'),
            color=alt.Color('type:N', title='Alert Type')
        )
        .properties(title="Alerts Frequency by Hour")
    )
    st.altair_chart(chart, use_container_width=True)

    # --- Display raw alert data ---
    st.subheader("🧾 Recent Alerts")
    st.dataframe(df)
else:
    st.info("ℹ️ No alerts to display yet. Waiting for activity...")

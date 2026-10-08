"""
Event Logs Module
-----------------
Contains both:
✅ Streamlit UI to view alert history
✅ log_event() function for backend logging
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import json
import os


LOG_FILE = os.path.join(os.path.dirname(__file__), "event_log.json")

# API authentication token from environment variable
API_TOKEN = os.getenv("SMART_CCTV_ADMIN_TOKEN", "")


# 🧠 1️⃣ Function for backend logging (used by main.py)
def log_event(message: str):
    """Log event to a local JSON file (used for internal system logging)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "message": message}

    # Append to JSON log
    logs = []

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    logs.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"[LOGGED] {timestamp}: {message}")


# 🧭 2️⃣ Streamlit UI — only runs when launched with `streamlit run`
if __name__ == "__main__" or "streamlit" in os.environ.get("STREAMLIT_SERVER_STATUS", ""):
    st.set_page_config(page_title="📊 Alert History", layout="wide")
    st.title("📊 Alert History")

    try:
        response = requests.get(
            "http://localhost:5000/alerts",
            headers={"Authorization": API_TOKEN},
            timeout=5
        )

        if response.status_code == 200:
            alerts = response.json()

            if isinstance(alerts, dict):
                df = pd.DataFrame([alerts])
            elif isinstance(alerts, list):
                df = pd.DataFrame(alerts)
            else:
                st.warning("⚠️ Unexpected alert format from server.")
                df = pd.DataFrame()

        else:
            st.error(f"🚨 Server returned status {response.status_code}")
            df = pd.DataFrame()

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to Flask API at http://localhost:5000/alerts")
        df = pd.DataFrame()

    except Exception as e:
        st.error(f"⚠️ Error fetching alerts: {e}")
        df = pd.DataFrame()

    # Display DataFrame or info message
    if not df.empty:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )
            df = df.sort_values(
                by="timestamp",
                ascending=False
            )

        st.dataframe(df, use_container_width=True)

    else:
        st.info("No alerts available yet.")
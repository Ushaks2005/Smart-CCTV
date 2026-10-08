# ===============================================
# SMART CCTV Main Controller (Windows Safe)
# ===============================================

import sys, io
import threading
import logging
import os
import time

# Fix for Windows console emoji/Unicode encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure local imports work no matter where we run from
sys.path.append(os.path.dirname(__file__))

# ===============================================
# ALERT SYSTEM IMPORTS
# ===============================================
from alert.sms.twilio_alert import send_twilio_alert
from alert.email.smtp_alert import send_email_alert
from alert.dashboard.websocket_alert import send_websocket_alert

# ===============================================
# CAMERA IMPORTS
# ===============================================
from camera.rtsp_stream import start_rtsp_stream
from camera.webcam_stream import start_web_stream

# ===============================================
# DASHBOARD & LOGGING IMPORTS
# ===============================================
from dashboard.flask_api.routes import start_flask_api
from dashboard.streamlit_app.live_stream import launch_streamlit_dashboard
from dashboard.streamlit_app.event_logs import log_event

# ===============================================
# DETECTION MODULES
# ===============================================
from detection.crime.detector import detect_crime
from detection.health.detector import detect_health

# ===============================================
# PREDICTION MODULES
# ===============================================
from model.crime.predict_crime import predict_crime_event
from model.health.predict_health import predict_health_event

# ===============================================
# TRACKING MODULES
# ===============================================
from trackking.deep_sort.tracker import track_objects
from trackking.object_id.assigner import assign_ids

# ===============================================
# LOGGING CONFIGURATION
# ===============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("smart_cctv.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ===============================================
# THREAD RUNNER
# ===============================================
def run_in_thread(target, name):
    """Runs any target safely inside a daemon thread."""
    def wrapper():
        try:
            logging.info(f"▶️ Starting {name}...")
            target()
            logging.info(f"✅ {name} is running.")
        except Exception as e:
            logging.error(f"❌ {name} failed: {e}")
            log_event(f"{name} failed: {e}")

    t = threading.Thread(target=wrapper, name=name, daemon=True)
    t.start()
    return t

# ===============================================
#  MODULE STARTERS
# ===============================================
def start_alerts():
    try:
        send_twilio_alert("SMART_CCTV started successfully.")
        send_email_alert("System Boot", "SMART_CCTV is live and running.")
        run_in_thread(send_websocket_alert, "WebSocket Alert")
        log_event("Alerts initialized.")
    except Exception as e:
        logging.error(f"Alert system failed: {e}")
        log_event(f"Alert system failed: {e}")

def start_camera():
    run_in_thread(start_rtsp_stream, "RTSP Stream")
    run_in_thread(start_web_stream, "Webcam Stream")

def start_detection():
    run_in_thread(detect_crime, "Crime Detection")
    run_in_thread(detect_health, "Health Detection")

def start_prediction():
    run_in_thread(predict_crime_event, "Crime Prediction")
    run_in_thread(predict_health_event, "Health Prediction")

def start_tracking():
    run_in_thread(track_objects, "Object Tracking")
    run_in_thread(assign_ids, "Object ID Assigner")

def start_dashboard():
    run_in_thread(start_flask_api, "Flask API Server")
    run_in_thread(launch_streamlit_dashboard, "Streamlit Dashboard")

# ===============================================
# SAFE START WRAPPER
# ===============================================
def safe_start(name, func):
    try:
        logging.info(f"🚀 Starting module: {name}")
        func()
    except Exception as e:
        logging.error(f"{name} failed to start: {e}")
        log_event(f"{name} failed to start: {e}")

# ===============================================
# MAIN ENTRY POINT
# ===============================================
def main():
    logging.info("🚀 SMART_CCTV System Booting...\n")
    log_event("SMART_CCTV initializing...")

    # Sequential startup
    safe_start("Alerts", start_alerts)
    safe_start("Camera", start_camera)
    safe_start("Detection", start_detection)
    safe_start("Prediction", start_prediction)
    safe_start("Tracking", start_tracking)
    safe_start("Dashboard", start_dashboard)

    log_event("SMART_CCTV initialized successfully.")
    logging.info("✅ SMART_CCTV System is Live and Monitoring.\n")

    # 🕒 Keep main thread alive safely for Windows
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("🛑 SMART_CCTV shutting down by user.")
        log_event("SMART_CCTV stopped by user.")
        os._exit(0)

# ===============================================
# RUN THE SYSTEM
# ===============================================
if __name__ == "__main__":
    main()

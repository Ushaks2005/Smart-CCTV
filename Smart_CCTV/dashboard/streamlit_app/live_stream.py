import streamlit as st
import cv2
import numpy as np

def launch_streamlit_dashboard():
    st.set_page_config(page_title="Live CCTV Stream", layout="wide")
    st.title("📹 Live CCTV Feed")

    st.markdown("Real-time video stream from your camera or RTSP feed.")

    # Select source
    source_type = st.selectbox("Select Source", ["Webcam", "RTSP URL"])

    if source_type == "Webcam":
        camera_index = st.number_input("Camera Index", 0, 10, 0)
        cap = cv2.VideoCapture(camera_index)
    else:
        rtsp_url = st.text_input("Enter RTSP URL", "rtsp://username:password@ip_address:port/stream")
        cap = cv2.VideoCapture(rtsp_url)

    stframe = st.empty()

    if st.button("Start Stream"):
        if not cap.isOpened():
            st.error("Failed to open camera or RTSP stream.")
        else:
            st.info("Stream started. Click Stop to end.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame.")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB", use_container_width=True)

                # Add a stop button
                if st.button("Stop Stream"):
                    break

            cap.release()
            st.success("Stream stopped successfully.")

if __name__ == "__main__":
    launch_streamlit_dashboard()

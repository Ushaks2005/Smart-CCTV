import cv2
import logging

def start_rtsp_stream(rtsp_url="rtsp://username:password@192.168.0.100:554/stream1"):
    """
    Starts capturing video from an RTSP camera stream.
    """
    try:
        logging.info("🎥 Connecting to RTSP camera...")
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise ValueError("Cannot connect to RTSP stream. Check URL or camera connection.")

        logging.info("✅ RTSP stream connected successfully.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ Failed to read frame from RTSP stream.")
                break

            # (Optional) Show the stream window for testing
            cv2.imshow("RTSP Stream", frame)

            # Press 'q' to exit manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"❌ RTSP stream error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("🔌 RTSP stream stopped.")

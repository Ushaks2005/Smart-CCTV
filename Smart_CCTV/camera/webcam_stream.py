import cv2

def start_web_stream(camera_index=0):
    """Start webcam stream using OpenCV."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam stream started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        cv2.imshow("Smart CCTV - Webcam Stream", frame)

        # Quit when user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Stream stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

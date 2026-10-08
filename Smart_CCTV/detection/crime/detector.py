import cv2
import numpy as np
from ultralytics import YOLO

# Load model (you can also use 'yolov8n.pt' for smaller model)
model = YOLO("yolov8n.pt")  # Automatically downloads if not present

def detect_crime(frame):
    """
    Detects suspicious or dangerous activity in a video frame.
    Returns processed frame and an alert flag.
    """
    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model.predict(source=img_rgb, stream=False, verbose=False)

    alert = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Simple "crime" keywords
            if name.lower() in ["knife", "gun", "weapon", "person with weapon"]:
                alert = True

    return frame, alert

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (you can later fine-tune on health/fall datasets)
model = YOLO("yolov8n.pt")  # Small, fast model
model.to("cpu")  # Force CPU use for compatibility

def detect_health(frame):
    """
    Detects health-related incidents such as falls or fainting.
    Returns processed frame and alert flag.
    """

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, stream=False, verbose=False)

    alert = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            # Very basic heuristic for a fall (horizontal person)
            if name.lower() == "person":
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height
                if aspect_ratio > 1.8:  # person is lying down
                    alert = True

    return frame, alert

import cv2
import os

def extract_frames(video_path, out_dir, every_n=5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            fname = f"frame_{saved:03d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1
        count += 1
    cap.release()
    print(f"✅ Extracted {saved} frames from {os.path.basename(video_path)}")
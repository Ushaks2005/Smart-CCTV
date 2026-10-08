import os

def validate_labels(video_dir, allowed_labels):
    errors = []
    for fname in os.listdir(video_dir):
        if not fname.endswith(".mp4"):
            continue
        label = fname.split("_")[0]
        if label not in allowed_labels:
            errors.append(fname)
    if errors:
        print("❌ Invalid labels found:")
        for e in errors:
            print(f" - {e}")
    else:
        print("✅ All labels are valid.")
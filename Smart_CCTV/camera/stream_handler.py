import yaml
from camera.webcam_stream import get_webcam_stream
from camera.rtsp_stream import get_rtsp_stream

def get_stream():
    with open("camera/camera_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    source = config.get("default_source", "webcam")

    if source == "webcam":
        index = config["webcam"]["index"]
        return get_webcam_stream(index)
    elif source == "rtsp":
        url = config["rtsp"]["url"]
        return get_rtsp_stream(url)
    else:
        raise ValueError(f"Unsupported camera source: {source}")

# detection/fire/postprocess.py
def threshold_alert(result, threshold=0.6):
    if result["label"] == "fire" and result["score"] >= threshold:
        return {"alert": True, "message": f"FIRE detected (score={result['score']:.3f})"}
    return {"alert": False, "message": f"No fire ({result['label']}, score={result['score']:.3f})"}

def format_output(result):
    return result["message"]
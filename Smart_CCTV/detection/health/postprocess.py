# detection/health/postprocess.py
def flag_abnormal(result, threshold=0.5):
    return {"abnormal": result["label"] == "abnormal" and result["score"] >= threshold, "score": result["score"]}

def format_output(result):
    if result.get("abnormal", False):
        return f"ABNORMAL detected (score={result['score']:.3f})"
    return f"Normal (score={result['score']:.3f})"
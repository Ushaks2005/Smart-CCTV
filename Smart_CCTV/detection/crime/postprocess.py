# detection/crime/postprocess.py
def filter_by_score(result, threshold=0.5):
    return result if result["score"] >= threshold else {"label": "unknown", "score": result["score"]}

def format_output(result):
    return f"{result['label']} (score={result['score']:.3f})"
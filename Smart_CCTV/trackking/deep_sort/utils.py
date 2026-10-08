# trackking/deep_sort/utils.py
import numpy as np

def iou(boxA, boxB):
    # boxes in [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = inter_w * inter_h
    boxA_area = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    boxB_area = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = boxA_area + boxB_area - inter
    return inter / union if union > 0 else 0.0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        suppressed = []
        for j in rest:
            if iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.append(j)
        idxs = np.array([x for x in rest if x not in suppressed])
    return keep
import os
import random
from sklearn.model_selection import train_test_split

def collect_image_paths(root, exts=(".jpg", ".jpeg", ".png")):
    items = []
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(exts):
                rel_path = os.path.join(class_name, fname).replace("\\", "/")
                items.append((rel_path, class_name))
    return items

def stratified_split(items, train_ratio, val_ratio, test_ratio, seed=42):
    paths = [p for p, c in items]
    labels = [c for p, c in items]
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, train_size=train_ratio, stratify=labels, random_state=seed
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths, temp_labels, train_size=val_ratio_adjusted, stratify=temp_labels, random_state=seed
    )
    return train_paths, val_paths, test_paths

def write_list(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
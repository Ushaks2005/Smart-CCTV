# shared/utils.py
"""
Collection of small helper utilities commonly used across modules:
- get_device, set_seed, load_yaml, ensure_dir, accuracy
Drop-in, copy-paste ready.
"""

import os
import random
import yaml
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    preds: raw logits or probabilities tensor of shape (N, C) or predicted labels (N,)
    labels: ground truth labels (N,)
    """
    if preds.dim() == 2:
        preds = preds.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    total = labels.size(0)
    return correct / total if total else 0.0

def topk_predictions(logits: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (topk_vals, topk_idxs)
    """
    return torch.topk(logits, k=k, dim=1)

def read_text_lines(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def write_text_lines(path: str, lines: Iterable[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for l in lines:
            f.write(str(l).strip() + "\n")
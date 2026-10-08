# shared/base_model.py
"""
Lightweight BaseModel wrapper for PyTorch models.
Drop-in, copy-paste ready. Provides:
- device handling
- save / load state_dict
- simple train / eval step helpers
- optional metric logging via provided logger function
"""

import os
import time
from typing import Callable, Optional, Dict, Any

import torch
import torch.nn as nn

class BaseModel:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        model: a torch.nn.Module (uninitialized or ready)
        device: torch.device or None (auto-selects cuda if available)
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self._is_training = False

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Save state_dict and optional extra metadata.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"state_dict": self.model.state_dict()}
        if extra:
            payload["meta"] = extra
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[torch.device] = None, strict: bool = True) -> Dict[str, Any]:
        """
        Load state_dict. Returns loaded payload (may contain 'meta').
        """
        map_location = map_location or self.device
        payload = torch.load(path, map_location=map_location)
        sd = payload.get("state_dict", payload)
        self.model.load_state_dict(sd, strict=strict)
        return payload

    def train_one_epoch(
        self,
        dataloader,
        criterion,
        optimizer,
        *,
        scheduler=None,
        gradient_clip: Optional[float] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform one training epoch. Returns metrics dict: {'loss':..., 'acc':...}
        """
        self.model.train()
        self._is_training = True
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for i, (imgs, labels) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += float(loss.item()) * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

        epoch_loss = running_loss / total if total else 0.0
        epoch_acc = correct / total if total else 0.0
        dur = time.time() - start
        if log_fn:
            log_fn(f"train loss={epoch_loss:.4f} acc={epoch_acc:.4f} time={dur:.1f}s")
        self._is_training = False
        return {"loss": epoch_loss, "acc": epoch_acc, "time": dur}

    def eval_on_loader(
        self,
        dataloader,
        criterion=None,
        *,
        log_fn: Optional[Callable[[str], None]] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataloader. If criterion provided, returns loss too.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                if criterion:
                    loss = criterion(outputs, labels)
                    running_loss += float(loss.item()) * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += labels.size(0)

        eval_loss = running_loss / total if (criterion and total) else None
        eval_acc = correct / total if total else 0.0
        dur = time.time() - start
        if log_fn:
            if eval_loss is not None:
                log_fn(f"eval loss={eval_loss:.4f} acc={eval_acc:.4f} time={dur:.1f}s")
            else:
                log_fn(f"eval acc={eval_acc:.4f} time={dur:.1f}s")
        return {"loss": eval_loss, "acc": eval_acc, "time": dur}

    def to_device(self, x):
        """
        Move tensor or nested structure to model device.
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, (list, tuple)):
            return type(x)(self.to_device(i) for i in x)
        if isinstance(x, dict):
            return {k: self.to_device(v) for k, v in x.items()}
        return x

    def state_dict(self):
        return self.model.state_dict()

    def parameters(self):
        return self.model.parameters()
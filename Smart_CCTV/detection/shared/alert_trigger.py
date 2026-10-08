# shared/alert_trigger.py
"""
Lightweight alert trigger used by detection modules.
Drop-in, copy-paste-ready. No external dependencies required.
"""

import time
import threading
import hashlib
import json
from collections import defaultdict
from typing import Callable, Dict, Optional

DEFAULT_DEBOUNCE_SECONDS = 5.0
DEFAULT_RATE_LIMIT_PER_MIN = 10

class AlertTrigger:
    """
    Thread-safe alert dispatcher with debounce and simple rate-limiting.
    Usage:
      at = AlertTrigger()
      at.register_handler(at.print_handler)
      at.trigger(label="fire", score=0.92, source="camera_1")
    """

    def __init__(
        self,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
    ):
        self.debounce_seconds = float(debounce_seconds)
        self.rate_limit_per_min = int(rate_limit_per_min)
        self._last_sent: Dict[str, float] = {}            # key -> timestamp
        self._counts: Dict[str, int] = defaultdict(int)   # key -> count in current window
        self._window_start = time.time()
        self._handlers: Dict[str, Callable[[dict], None]] = {}
        self._lock = threading.Lock()

    # --- Handlers -------------------------------------------------------
    def register_handler(self, name: str, fn: Callable[[dict], None]) -> None:
        self._handlers[name] = fn

    def unregister_handler(self, name: str) -> None:
        self._handlers.pop(name, None)

    @staticmethod
    def print_handler(payload: dict) -> None:
        print("[ALERT]", json.dumps(payload, ensure_ascii=False))

    # --- Utilities ------------------------------------------------------
    def _make_key(self, label: str, source: Optional[str]) -> str:
        base = f"{label}|{source or 'global'}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _reset_window_if_needed(self) -> None:
        now = time.time()
        if now - self._window_start >= 60.0:
            self._window_start = now
            self._counts.clear()

    # --- Core API ------------------------------------------------------
    def trigger(
        self,
        label: str,
        score: float,
        source: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[dict] = None,
        severity: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Attempt to send an alert. Returns True if dispatched to handlers.
        - debounce_seconds prevents repeated identical alerts from same source.
        - rate_limit_per_min globally limits alerts per unique key per minute.
        - force bypasses debounce and rate-limit.
        """
        now = timestamp or time.time()
        key = self._make_key(label, source)
        payload = {
            "label": label,
            "score": float(score),
            "source": source,
            "timestamp": now,
            "severity": severity or ("high" if score >= 0.9 else "medium" if score >= 0.6 else "low"),
            "metadata": metadata or {},
        }

        with self._lock:
            if not force:
                # debounce check
                last = self._last_sent.get(key)
                if last and (now - last) < self.debounce_seconds:
                    return False
                # rate-limit check (per-key per minute)
                self._reset_window_if_needed()
                if self._counts[key] >= self.rate_limit_per_min:
                    return False
                self._counts[key] += 1

            # record send time
            self._last_sent[key] = now

        # dispatch to handlers outside lock
        for h in list(self._handlers.values()):
            try:
                h(payload)
            except Exception:
                continue

        return True

    # --- Convenience helpers -------------------------------------------
    def send_webhook(self, url: str, payload: dict) -> None:
        """
        Minimal webhook sender using urllib to avoid extra deps.
        It runs in a background thread to avoid blocking detection loop.
        """
        import urllib.request

        def _worker(u, p):
            try:
                data = json.dumps(p).encode("utf-8")
                req = urllib.request.Request(u, data=data, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass

        threading.Thread(target=_worker, args=(url, payload), daemon=True).start()

    def send_via_handlers(self, payload: dict, webhook_url: Optional[str] = None) -> None:
        if webhook_url:
            self.send_webhook(webhook_url, payload)
        for h in list(self._handlers.values()):
            try:
                h(payload)
            except Exception:
                pass

# --- Example ready-to-copy usage --------------------------------------
if __name__ == "__main__":
    at = AlertTrigger(debounce_seconds=3.0, rate_limit_per_min=5)
    at.register_handler("print", AlertTrigger.print_handler)

    # Example: detection modules call this
    dispatched = at.trigger(label="fire", score=0.95, source="cam_12", metadata={"frame":"frame_001.jpg"})
    print("dispatched:", dispatched)
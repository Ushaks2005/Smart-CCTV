# trackking/object_id/history_buffer.py
from collections import defaultdict, deque

class HistoryBuffer:
    def __init__(self, maxlen=50):
        self.buffer = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, track_id, bbox):
        self.buffer[track_id].append(bbox)

    def get(self, track_id):
        return list(self.buffer.get(track_id, []))

    def remove(self, track_id):
        if track_id in self.buffer:
            del self.buffer[track_id]

    def recent(self):
        return {k: list(v) for k, v in self.buffer.items()}
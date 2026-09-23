"""Append-only JSONL store + a process-global recorder."""
from __future__ import annotations

import json
import os
import threading


class Recorder:
    def __init__(self, run_id: str = "run", out: str | None = None):
        self.run_id = run_id
        self.out = out
        self._lock = threading.Lock()
        self.events: list[dict] = []      # in-memory mirror (for live / offline analysis)
        self._fh = None
        if out:
            d = os.path.dirname(os.path.abspath(out))
            if d:
                os.makedirs(d, exist_ok=True)
            self._fh = open(out, "a", encoding="utf-8")

    def emit(self, event) -> None:
        rec = event.to_json() if hasattr(event, "to_json") else dict(event)
        with self._lock:
            self.events.append(rec)
            if self._fh:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


_recorder: Recorder | None = None


def init(run_id: str = "run", out: str | None = None) -> Recorder:
    global _recorder
    _recorder = Recorder(run_id, out)
    return _recorder


def recorder() -> Recorder:
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder

"""Tiny JSON file cache so repeated runs on the same day don't re-hit public APIs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from nasdaq_agent.context import get_ctx


def _path(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    root = get_ctx().settings.output_root / ".cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}.json"


def cached(key: str, ttl_s: float, produce: Callable[[], Any]) -> tuple[Any, bool]:
    """Return ``(value, hit)``. ``produce`` must return JSON-serialisable data."""
    path = _path(key)
    if ttl_s > 0 and path.exists():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - entry["ts"] < ttl_s:
                return entry["value"], True
        except (ValueError, KeyError):
            pass
    value = produce()
    path.write_text(json.dumps({"ts": time.time(), "key": key, "value": value}, default=str), encoding="utf-8")
    return value, False

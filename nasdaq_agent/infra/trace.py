"""Structured run trace: one JSON line per event in ``outputs/run_*/trace.jsonl``."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def write(run_dir: Path, event: str, **data: Any) -> None:
    line = json.dumps({"ts": round(time.time(), 3), "event": event, **data}, default=str, ensure_ascii=False)
    with _lock, (run_dir / "trace.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


@contextmanager
def run_logging(run_dir: Path, level: int = logging.INFO):
    from nasdaq_agent.context import get_ctx
    class RunFilter(logging.Filter):
        def filter(self, record):
            try:
                return get_ctx().run_dir == run_dir
            except RuntimeError:
                return False
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.FileHandler(run_dir / "agent.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RunFilter())
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "yfinance", "peewee", "urllib3", "matplotlib", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()

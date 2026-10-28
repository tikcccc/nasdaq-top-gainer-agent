"""Historical date/stock/recipient log reader; no longer consulted for delivery."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()


def _key(trade_date: str, symbol: str, recipients: list[str]) -> str:
    return f"{trade_date}|{symbol.upper()}|{','.join(sorted(r.lower() for r in recipients))}"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def already_sent(path: Path, trade_date: str, symbol: str, recipients: list[str]) -> dict | None:
    return _load(path).get(_key(trade_date, symbol, recipients))


def record(path: Path, trade_date: str, symbol: str, recipients: list[str], info: dict) -> None:
    with _lock:
        data = _load(path)
        data[_key(trade_date, symbol, recipients)] = {**info, "sent_at": datetime.now().isoformat(timespec="seconds")}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

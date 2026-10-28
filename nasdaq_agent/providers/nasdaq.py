"""Nasdaq.com public JSON endpoints (no key). Used as the fallback source.

Note: the screener endpoint carries no timestamp and has been observed to lag a
session behind, so its rows are freshness-checked by the caller.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from nasdaq_agent.data.screener_clean import parse_number
from nasdaq_agent.infra import http
from nasdaq_agent.infra.errors import ProviderError

NAME = "nasdaq"
BASE = "https://api.nasdaq.com/api"
HEADERS = {"Accept": "application/json", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}


def _data(payload: Any, what: str) -> dict:
    data = (payload or {}).get("data")
    if not data:
        status = (payload or {}).get("status") or {}
        raise ProviderError(NAME, f"{what}: empty response {status.get('bCodeMessage') or ''}")
    return data


def screener_gainers() -> list[dict[str, Any]]:
    payload = http.get_json(
        f"{BASE}/screener/stocks",
        params={"tableonly": "true", "exchange": "nasdaq", "limit": 5000, "download": "true"},
        headers=HEADERS,
    )
    rows = _data(payload, "screener").get("rows") or []
    if len(rows) < 500:
        raise ProviderError(NAME, f"screener returned only {len(rows)} rows")
    out = []
    for r in rows:
        pct = parse_number(r.get("pctchange"))
        if pct is None:
            continue
        out.append({
            "symbol": r.get("symbol"),
            "name": r.get("name", ""),
            "price": parse_number(r.get("lastsale")),
            "pct_change": pct,
            "net_change": parse_number(r.get("netchange")),
            "volume": int(parse_number(r.get("volume")) or 0),
            "market_cap": parse_number(r.get("marketCap")),
            "exchange": "NASDAQ",
            "market_time": None,
        })
    return out


def history(symbol: str, start: date, end: date, assetclass: str = "stocks") -> pd.DataFrame:
    payload = http.get_json(
        f"{BASE}/quote/{symbol}/historical",
        params={"assetclass": assetclass, "fromdate": start.isoformat(), "todate": end.isoformat(), "limit": 100},
        headers=HEADERS,
    )
    table = (_data(payload, f"history {symbol}").get("tradesTable") or {}).get("rows") or []
    if not table:
        raise ProviderError(NAME, f"no history rows for {symbol}")
    bars = [
        {
            "date": datetime.strptime(r["date"], "%m/%d/%Y").date(),
            "open": parse_number(r.get("open")),
            "high": parse_number(r.get("high")),
            "low": parse_number(r.get("low")),
            "close": parse_number(r.get("close")),
            "volume": parse_number(r.get("volume")),
        }
        for r in table
    ]
    return pd.DataFrame(bars)

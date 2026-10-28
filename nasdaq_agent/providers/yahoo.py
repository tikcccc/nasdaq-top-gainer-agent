"""Yahoo Finance via yfinance: screener (timestamped quotes), daily history, news."""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from nasdaq_agent.data.calendar import to_ny_date
from nasdaq_agent.infra.errors import ProviderError, StaleDataError

warnings.filterwarnings("ignore", module="yfinance")
import yfinance as yf  # noqa: E402

log = logging.getLogger(__name__)
NAME = "yahoo"
NASDAQ_EXCHANGES = ["NMS", "NCM", "NGM"]  # Global Select, Capital Market, Global Market

_retry = retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=10), reraise=True)


@_retry
def _screen(size: int) -> list[dict[str, Any]]:
    q = yf.EquityQuery
    query = q("and", [q("is-in", ["exchange", *NASDAQ_EXCHANGES]), q("gt", ["percentchange", 0])])
    result = yf.screen(query, sortField="percentchange", sortAsc=False, size=size)
    return result.get("quotes", [])


def screener_gainers(trade_date: date, is_intraday: bool, size: int = 250) -> list[dict[str, Any]]:
    quotes = _screen(size)
    if not quotes:
        raise ProviderError(NAME, "screener returned no quotes")
    rows = []
    for item in quotes:
        ts = item.get("regularMarketTime")
        rows.append({
            "symbol": item.get("symbol"),
            "name": item.get("longName") or item.get("shortName") or "",
            "price": item.get("regularMarketPrice"),
            "pct_change": item.get("regularMarketChangePercent"),
            "net_change": item.get("regularMarketChange"),
            "volume": item.get("regularMarketVolume"),
            "market_cap": item.get("marketCap"),
            "exchange": item.get("exchange"),
            "quote_type": item.get("quoteType"),
            "market_time": to_ny_date(ts).isoformat() if ts else None,
        })
    # Freshness: the leading quotes must belong to the trade date.
    dated = [r["market_time"] for r in rows[:20] if r["market_time"]]
    if not is_intraday and dated:
        fresh = sum(d == trade_date.isoformat() for d in dated)
        if fresh < len(dated) / 2:
            raise StaleDataError(NAME, f"quotes are from {max(set(dated), key=dated.count)}, expected {trade_date}")
    return rows


@_retry
def _history(symbol: str, start: date, end: date) -> pd.DataFrame:
    return yf.Ticker(symbol).history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=False, actions=True
    )


def history(symbol: str, start: date, end: date) -> tuple[pd.DataFrame, dict, dict]:
    """Daily bars (split-adjusted, not dividend-adjusted) plus split and dividend events."""
    df = _history(symbol, start, end)
    if df is None or df.empty:
        raise ProviderError(NAME, f"no history for {symbol}")
    df = df.copy()
    df.index = [ts.tz_convert("America/New_York").date() if ts.tzinfo else ts.date() for ts in df.index]
    splits = {d: float(v) for d, v in df.get("Stock Splits", pd.Series(dtype=float)).items() if v}
    dividends = {d: float(v) for d, v in df.get("Dividends", pd.Series(dtype=float)).items() if v}
    bars = df.rename_axis("date").reset_index()[["date", "Open", "High", "Low", "Close", "Volume"]]
    return bars, splits, dividends


@_retry
def news(symbol: str, count: int = 10) -> list[dict[str, Any]]:
    items = yf.Search(symbol, news_count=count, max_results=1).news or []
    out = []
    for item in items:
        ts = item.get("providerPublishTime")
        out.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "published": pd.Timestamp(ts, unit="s", tz="UTC").isoformat() if ts else None,
            "link": item.get("link"),
            "related": item.get("relatedTickers") or [],
        })
    return out

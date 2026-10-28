"""Numbered, clickable references so every claim in the email can be checked.

The agent cites evidence by id: ``[d1]``…``[d7]`` for data (screener, price
histories, listing, filings) and ``[n1]``… for headlines. The renderer turns the
ids into academic-style numbers ``[1]``, ``[2]`` in order of first use, links each
one to its source page, and lists them under "Sources" at the end of the email.
Only data *sources* are referenced: models (the LLM, Jev) are not sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from nasdaq_agent.reporting.placeholders import human_date
from nasdaq_agent.reporting.announcements import matched
from nasdaq_agent.reporting.dates import news_date

# What each data id means - shown to the agent in the system prompt.
DATA_REFS = {
    "d1": "the top-gainers screener the stock was selected from",
    "d2": "the stock's daily price history for the window",
    "d3": "S&P 500 daily history for the same window",
    "d4": "NASDAQ Composite daily history for the same window",
    "d5": "the stock's exchange listing / company profile",
    "d6": "the company's regulatory filings (SEC EDGAR)",
    "d7": "the excluded unfiltered #1 (only when it differs from the analysed stock)",
}


@dataclass
class Reference:
    id: str
    title: str
    publisher: str
    url: str
    date: str = ""
    number: int = 0


def _yahoo_history(symbol: str) -> str:
    return f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/history/"


def _nasdaq_history(symbol: str, asset: str = "stocks") -> str:
    return f"https://www.nasdaq.com/market-activity/{asset}/{symbol.lower()}/historical"


def available(state: dict) -> dict[str, Reference]:
    """Every reference this run can support, keyed by id."""
    selected = state.get("selected") or {}
    sources = state.get("sources") or {}
    datasets = state.get("datasets") or {}
    symbol = selected.get("symbol")
    trade_date = (state.get("session") or {}).get("trade_date")
    when = human_date(trade_date) if trade_date else ""
    window = ""
    meta = datasets.get(selected.get("dataset_id", "")) or {}
    if meta:
        window = f"{human_date(meta['first_date'])} – {human_date(meta['last_date'])}"

    refs: dict[str, Reference] = {}
    if sources.get("screener") == "nasdaq":
        refs["d1"] = Reference("d1", "NASDAQ stock screener (percent change)", "Nasdaq.com",
                               "https://www.nasdaq.com/market-activity/stocks/screener", when)
    else:
        refs["d1"] = Reference("d1", "Top stock gainers (NASDAQ listings)", "Yahoo Finance",
                               "https://finance.yahoo.com/markets/stocks/gainers/", when)
    if symbol:
        history_source = meta.get('source') or sources.get("history")
        if history_source == "nasdaq":
            refs["d2"] = Reference("d2", f"{symbol} historical prices", "Nasdaq.com", _nasdaq_history(symbol), window)
        elif history_source == 'yahoo':
            refs["d2"] = Reference("d2", f"{symbol} historical prices", "Yahoo Finance", _yahoo_history(symbol), window)
        else:
            refs["d2"] = Reference("d2", f"{symbol} historical prices", history_source or 'Snapshot data', '', window)
        refs["d5"] = Reference("d5", f"{selected.get('name') or symbol} ({symbol}) quote and company profile",
                               "Nasdaq.com", f"https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}")
        refs["d6"] = Reference("d6", f"{symbol} company filings", "SEC EDGAR",
                               f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={quote(symbol)}&owner=include")
    for ref_id, index, name, proxy in (("d3", "^GSPC", "S&P 500", "spy"), ("d4", "^IXIC", "NASDAQ Composite", "qqq")):
        bench = next((m for m in datasets.values() if (m.get("proxy_for") or m["symbol"]) == index), None)
        if bench and bench.get("proxy_for"):
            refs[ref_id] = Reference(ref_id, f"{name} proxy ({proxy.upper()} ETF) historical prices", "Nasdaq.com",
                                     _nasdaq_history(proxy, "etf"), window)
        elif bench or symbol:
            provider = (bench or {}).get('source') or sources.get(f'benchmark {index}') or sources.get('history')
            refs[ref_id] = Reference(ref_id, f"{name} ({index}) historical prices",
                                    'Yahoo Finance' if provider == 'yahoo' else provider or 'Snapshot data',
                                    _yahoo_history(index) if provider == 'yahoo' else '', window)
    raw_top = state.get("raw_top") or {}
    if raw_top.get("symbol") and raw_top["symbol"] != symbol:
            refs["d7"] = Reference("d7", f"{raw_top['symbol']} quote (outside the selected scope)", "Nasdaq.com",
                               f"https://www.nasdaq.com/market-activity/stocks/{raw_top['symbol'].lower()}", when)
    for hid, h in (state.get("headlines") or {}).items():
        # A fetched headline is always citable; without a link the number is shown but not clickable.
        published = news_date(h.get('published'))
        date = human_date(published) if published else ""
        refs[hid] = Reference(hid, h["title"], h.get("publisher") or "News", h.get("link") or "", date)
    for ref, item in matched(state).items():
        refs[ref] = Reference(ref, item['title'], item['publisher'], item['url'], human_date(item['published']))
    return refs


def number(refs: dict[str, Reference], order: list[str]) -> list[Reference]:
    """Assign [1], [2]… in the given order (duplicates and unknown ids skipped)."""
    numbered: list[Reference] = []
    for ref_id in order:
        ref = refs.get(ref_id)
        if ref and not ref.number:
            ref.number = len(numbered) + 1
            numbered.append(ref)
    return numbered

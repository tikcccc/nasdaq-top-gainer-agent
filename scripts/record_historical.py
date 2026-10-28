"""Record public Yahoo chart responses for explicit historical symbols, never a historical ranking.

Re-recording requires a new destination so prior evidence cannot silently change.
Only daily OHLCV/events enter replay; present-day metadata is not model evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from curl_cffi import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from nasdaq_agent.data.calendar import _cal

EPISODES = [
    ("aapl_earnings", "AAPL", "2024-05-03", "earnings_buyback", "2024-05-02",
     "Apple reports second quarter results", "Apple Newsroom",
     "https://www.apple.com/newsroom/2024/05/apple-reports-second-quarter-results/"),
    ("nvda_earnings", "NVDA", "2024-05-23", "earnings", "2024-05-22",
     "NVIDIA Announces Financial Results for First Quarter Fiscal 2025", "NVIDIA Newsroom",
     "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2025"),
    ("nvda_split_dividend", "NVDA", "2024-06-11", "split_dividend", "2024-06-07",
     "NVIDIA files stock split amendment; split-adjusted trading expected June 10", "NVIDIA SEC filing (paraphrase)",
     "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm"),
    ("meta_earnings_decline", "META", "2022-02-03", "earnings_decline_renamed_ticker", "2022-02-02",
     "Meta Reports Fourth Quarter and Full Year 2021 Results", "Meta Investor Relations",
     "https://facebook2016ir.q4web.com/investor-news/press-release-details/2022/Meta-Reports-Fourth-Quarter-and-Full-Year-2021-Results/default.aspx"),
    ("amzn_earnings_decline", "AMZN", "2022-04-29", "earnings_decline", "2022-04-28",
     "Amazon.com Announces First Quarter Results", "Amazon Investor Relations",
     "https://ir.aboutamazon.com/news-release/news-release-details/2022/Amazon-com-Announces-First-Quarter-Results-f0188db95/"),
    ("msft_half_day", "MSFT", "2024-11-29", "holiday_and_early_close", None, None, None, None),
]


def decode(payload):
    result = payload["chart"]["result"][0]
    quotes = result["indicators"]["quote"][0]
    rows = []
    for i, ts in enumerate(result["timestamp"]):
        row = {"date": datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).date().isoformat()}
        row.update({key: quotes[key][i] for key in ("open", "high", "low", "close", "volume")})
        rows.append(row)
    return rows, result.get("events", {})


def reference(stock, benchmarks, symbol):
    """Independent arithmetic for stored gold: no application metrics implementation."""
    closes = [r["close"] for r in stock[-6:]]
    returns = [b/a-1 for a,b in zip(closes,closes[1:])]
    cumulative = closes[-1]/closes[0]-1
    normalized = [c/closes[0] for c in closes]
    slope = sum((i-2.5)*(c-statistics.mean(normalized)) for i,c in enumerate(normalized))/17.5
    peak = closes[0]
    drawdowns = []
    for c in closes:
        peak=max(peak,c)
        drawdowns.append(c/peak-1)
    bench = {key:values[-1]["close"]/values[-6]["close"]-1 for key,values in benchmarks.items()}
    vol=statistics.stdev(returns)
    return dict(symbol=symbol,days=5,daily_returns=returns,cumulative_return=cumulative,
                avg_daily_return=statistics.mean(returns),trend_slope=slope,
                trend="up" if cumulative>.02 and slope>0 else "down" if cumulative<-.02 and slope<0 else "sideways",
                volatility_daily=vol,volatility_annualized=vol*math.sqrt(252),max_drawdown=min(drawdowns),
                benchmark=bench,excess_return=cumulative-bench["GSPC"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=ROOT/"tests/fixtures/historical")
    args=p.parse_args()
    if args.destination.exists():
        p.error("destination already exists; preserve prior recordings and use a new path")
    raw_dir=args.destination/"raw"
    raw_dir.mkdir(parents=True)
    manifest={"version":1,"kind":"recorded_historical_symbol_windows","ranking_accuracy":None,
        "limitations":"Retrospective Yahoo history recorded at retrieval time, not a point-in-time archive or independent price source. Current symbols and later split adjustments may apply (META traded as FB in 2022). Explicit symbol windows, not reconstructed full NASDAQ rankings. Issuer-news titles curated from primary releases; dates have day precision and conservative end-of-day UTC timestamps. No claims of news exhaustiveness or event causality. The no-news episode means no curated news, not that none existed.",
        "pricing":"Yahoo quote close/OHLC: split-adjusted, not dividend-adjusted. Never use adjclose for price returns or apply splits twice.",
        "cases":[]}
    for cid,symbol,trade,regime,pub,title,publisher,link in EPISODES:
        end=date.fromisoformat(trade)
        start=end-timedelta(days=24)
        series={}
        for ticker in (symbol,"^GSPC","^IXIC"):
            url=f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker,safe='')}"
            params={"period1":int(datetime.combine(start,datetime.min.time(),ZoneInfo('America/New_York')).timestamp()),
                    "period2":int(datetime.combine(end+timedelta(days=1),datetime.min.time(),ZoneInfo('America/New_York')).timestamp()),
                    "interval":"1d","events":"div,splits"}
            response=requests.get(url,params=params,impersonate="chrome",timeout=30)
            response.raise_for_status()
            payload=response.json()
            rows,events=decode(payload)
            expected=[d.date().isoformat() for d in _cal().sessions_in_range(pd.Timestamp(start),pd.Timestamp(end))]
            if [r["date"] for r in rows] != expected or any(r["close"] is None or r["close"]<=0 for r in rows):
                raise ValueError(f"bad dates/prices for {cid} {ticker}")
            filename=f"{cid}_{ticker.lstrip('^')}.json"
            target=raw_dir/filename
            target.write_bytes(response.content)
            series[ticker]={"raw_path":"raw/"+filename,"sha256":hashlib.sha256(response.content).hexdigest(),
                "url":str(response.url),"retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
                "bars":rows,"events":events}
            print(cid,ticker,len(rows),flush=True)
        gold=reference(series[symbol]["bars"],{k:series['^'+k]["bars"] for k in ("GSPC","IXIC")},symbol)
        manifest["cases"].append(dict(id=cid,symbol=symbol,trade_date=trade,regime=regime,series=series,gold=gold,
            calendar_close_utc=_cal().session_close(pd.Timestamp(trade)).isoformat(),
            news=[dict(title=title,publisher=publisher,published=pub+"T23:59:59+00:00",publication_precision="day",link=link)] if title else []))
        (args.destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Saved {len(manifest['cases'])} historical episodes")


if __name__=="__main__":
    main()

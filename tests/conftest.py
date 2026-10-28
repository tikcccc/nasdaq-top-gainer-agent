"""Offline fixtures: synthetic market data, a scripted LLM and an isolated run context."""

from __future__ import annotations

import itertools
import os
from datetime import date

# Tests never read the developer's .env, and credentials exported in the shell are dropped,
# so no real address or key can reach a fixture, a run output or a failure message.
os.environ["NASDAQ_ENV_FILE"] = "none"
for _name in [n for n in os.environ if n.startswith(("SMTP_", "EMAIL_", "MIMO_", "DEEPSEEK_", "OPENAI_",
                                                      "TYPESAFE_", "JEV_", "LLM_", "LANGSMITH_", "LANGCHAIN_"))]:
    del os.environ[_name]
os.environ.update({"SMTP_USER": "agent@example.invalid", "SMTP_PASSWORD": "test-password",
                   "EMAIL_FROM": "agent@example.invalid",
                   **{f"{p}_{k}": v for p in ("MIMO", "DEEPSEEK") for k, v in (
                       ("API_KEY", "offline-test-key"), ("BASE_URL", f"https://{p.lower()}.invalid/v1"), ("MODEL", "offline"))}})
os.environ.pop("NASDAQ_API_TOKEN", None)

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from nasdaq_agent.config import load_settings
from nasdaq_agent.context import start_run
from nasdaq_agent.data.calendar import MarketSession, _cal
from nasdaq_agent.infra import breaker

TRADE_DATE = date(2026, 10, 23)
SESSION = MarketSession(trade_date=TRADE_DATE, is_intraday=False, previous_date=date(2026, 10, 22))


def sessions(n: int) -> list[date]:
    cal = _cal()
    days = cal.sessions_window(pd.Timestamp(TRADE_DATE), -n)
    return [d.date() for d in days]


def bars(closes: list[float]) -> pd.DataFrame:
    dates = sessions(len(closes))
    return pd.DataFrame({
        "date": dates, "Open": closes, "High": [c * 1.02 for c in closes], "Low": [c * 0.98 for c in closes],
        "Close": closes, "Volume": [1_000_000] * len(closes),
    })


STOCK = [4.0] * 10 + [3.98, 4.93, 4.84, 4.25, 12.02]
GSPC = [760.0] * 10 + [762.0, 765.0, 766.0, 763.96, 759.97]
IXIC = [24000.0] * 10 + [24100.0, 24200.0, 24250.0, 24180.0, 24010.0]

SCREENER = [
    {"symbol": "ADNWW", "name": "Aadi Warrants", "price": 0.29, "pct_change": 2800.0, "net_change": 0.28,
     "volume": 50_000, "market_cap": None, "exchange": "NCM", "quote_type": "EQUITY", "market_time": "2026-10-23"},
    {"symbol": "AIOK", "name": "AI OKTO CORP.", "price": 12.02, "pct_change": 182.82, "net_change": 7.77,
     "volume": 41_847_594, "market_cap": 51_925_392, "exchange": "NCM", "quote_type": "EQUITY", "market_time": "2026-10-23"},
    {"symbol": "CLRO", "name": "ClearOne, Inc.", "price": 3.25, "pct_change": 69.27, "net_change": 1.33,
     "volume": 35_102_081, "market_cap": 8_695_089, "exchange": "NCM", "quote_type": "EQUITY", "market_time": "2026-10-23"},
]
NEWS = [
    {"title": "AI OKTO Corp. Declares Special Dividend Per Share", "publisher": "GlobeNewswire",
     "published": "2026-10-23T13:24:00+00:00", "link": "https://example.com/1"},
    {"title": "AI OKTO Announces the Sale of an LPG Carrier", "publisher": "GlobeNewswire",
     "published": "2026-10-23T12:00:00+00:00", "link": "https://example.com/2"},
]


@pytest.fixture
def settings(tmp_path):
    return load_settings(overrides={
        "run": {"output_dir": str(tmp_path / "outputs"), "deadline_s": 120},
        "email": {"dry_run": True, "to": ["test@example.com"]},
        "jev": {"enabled": False, "allow_rule_fallback": True},
        "sandbox": {"backend": "trusted_local"},  # only the reviewed scripts in these fixtures
        "providers": {
            "mimo": {"name": "mimo", "api_key": "offline-test-key", "base_url": "https://mimo.invalid/v1", "model": "offline"},
            "deepseek": {"name": "deepseek", "api_key": "offline-test-key", "base_url": "https://deepseek.invalid/v1", "model": "offline"},
        },
    })


@pytest.fixture(scope="session")
def history_archive(tmp_path_factory):
    """Optional local historical evidence; excluded from the public repository."""
    import zipfile
    from pathlib import Path
    path = Path(__file__).parents[1] / "docs/evaluation/history/history.zip"
    if not path.is_file():
        pytest.skip("local historical evaluation archive is not distributed")
    target = tmp_path_factory.mktemp("history")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    return target


@pytest.fixture
def ctx(settings):
    breaker.reset(3)
    return start_run(settings, "test")


@pytest.fixture
def offline_market(monkeypatch):
    """Patch every provider and the clock so tools run without network."""
    from nasdaq_agent.providers import google_news, nasdaq, yahoo
    from nasdaq_agent.tools import gainers, history

    series = {"AIOK": STOCK, "^GSPC": GSPC, "^IXIC": IXIC, "CLRO": [2.0] * 14 + [3.25]}

    def fake_history(symbol, start, end):
        return bars(series[symbol]), {}, {}

    monkeypatch.setattr(gainers, "market_session", lambda: SESSION)
    monkeypatch.setattr(history, "market_session", lambda: SESSION)
    monkeypatch.setattr(yahoo, "screener_gainers", lambda trade_date, is_intraday: [dict(r) for r in SCREENER])
    monkeypatch.setattr(yahoo, "history", fake_history)
    monkeypatch.setattr(yahoo, "news", lambda symbol, count=10: list(NEWS))
    monkeypatch.setattr(nasdaq, "screener_gainers", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    # The second vendor agrees with Yahoo unless a test says otherwise.
    monkeypatch.setattr(nasdaq, "history", lambda symbol, start, end, assetclass="stocks": bars(series[symbol]))
    monkeypatch.setattr(google_news, "news", lambda *a, **k: [])
    return series


class ScriptedLLM:
    """Stands in for the chat model: returns pre-written AI messages in order."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = 0
        self._ids = itertools.count(1)

    def invoke(self, messages):
        self.calls += 1
        if not self.steps:
            return AIMessage(content="done")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(messages)
        if isinstance(step, AIMessage):
            return step
        calls = [{"name": n, "args": a, "id": f"call_{next(self._ids)}", "type": "tool_call"} for n, a in step]
        return AIMessage(content="", tool_calls=calls)


ANALYSIS_CODE = '''
import numpy as np, math
import matplotlib.pyplot as plt
s = load("ds_AIOK")["close"].tail(6).reset_index(drop=True)
r = (s / s.shift(1) - 1).dropna().tolist()
bench = {k: float(load(f"ds_{k}")["close"].tail(6).iloc[-1] / load(f"ds_{k}")["close"].tail(6).iloc[0] - 1) for k in ("GSPC", "IXIC")}
cum = float(s.iloc[-1] / s.iloc[0] - 1)
vol = float(np.std(r, ddof=1))
save_result({"daily_returns": r, "avg_daily_return": float(np.mean(r)), "cumulative_return": cum,
             "trend": "up", "trend_slope": float(np.polyfit(range(6), s / s.iloc[0], 1)[0]),
             "volatility_daily": vol, "volatility_annualized": vol * math.sqrt(252),
             "max_drawdown": float((s / s.cummax() - 1).min()), "benchmark": bench,
             "excess_return": cum - bench["GSPC"]})
fig, ax = plt.subplots(); ax.plot(s / s.iloc[0] * 100); save_chart(fig)
'''

GOOD_NARRATIVE = (
    "**Why this stock.** {{symbol}} ({{company}}) was the {{selection_label}} on {{trade_date}}, up {{pct_change}}. "
    "It traded {{volume}} shares on the day and closed at {{price}} [d1]. The reported ranking uses the configured "
    "price and volume screen for common stocks.\n\n"
    "**How it traded.** {{path.sequence}} {{path.contribution}} {{path.benchmarks}} [d2][d3][d4]\n\n"
    "**What's behind it.** {{news.coverage}} A returned headline reports a special dividend [n1]. "
    "This identifies a company announcement in the fetched coverage, while its contribution to the stock's "
    "price movement has not been established by the supplied evidence."
)

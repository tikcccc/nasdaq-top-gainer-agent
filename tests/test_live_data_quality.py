"""Ranking and data-quality regressions, partly from real 2026-10-23 data
(tests/fixtures/live_20261023).

Yahoo's screener mixed quotes from earlier sessions into its top rows, Nasdaq's
screener still served the previous session, and three sources reported different
volumes for the selected stock. Price agreement between two vendors is kept apart
from the question of whether the selected stock is a confirmed #1."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nasdaq_agent.data import prices
from nasdaq_agent.data.calendar import MarketSession
from nasdaq_agent.data.cross_check import price_confirmation, quote_freshness
from nasdaq_agent.providers import nasdaq, yahoo
from nasdaq_agent.reporting import evidence_report
from nasdaq_agent.tools import gainers
from nasdaq_agent.tools.history import eligibility_check, get_price_history
from tests.conftest import SCREENER, SESSION, STOCK, bars

LIVE = Path(__file__).parent / "fixtures" / "live_20261023"
YAHOO = json.loads((LIVE / "yahoo_screener.json").read_text(encoding="utf-8"))
NASDAQ = json.loads((LIVE / "nasdaq.json").read_text(encoding="utf-8"))
D21, D22, D23 = date(2026, 10, 21), date(2026, 10, 22), date(2026, 10, 23)


def _payload(command) -> dict:
    return json.loads(command.update["messages"][0].content)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.assign(open=frame["close"], high=frame["close"], low=frame["close"])


def _history_from(table: dict):
    def history(symbol, *args, **kwargs):
        rows = table.get(symbol)
        if not isinstance(rows, list):
            raise RuntimeError(f"no history for {symbol}")
        return _frame(rows)
    return history


def _closes(*values, volume=1_000_000) -> list[dict]:
    """Bars for the 21st, 22nd and 23rd."""
    return [{"date": str(d), "close": c, "volume": volume} for d, c in zip((D21, D22, D23), values)]


@pytest.fixture
def market(monkeypatch):
    """Recorded screeners; Yahoo is the live source unless a test takes it down."""
    monkeypatch.setattr(gainers, "market_session", lambda: SESSION)
    monkeypatch.setattr(nasdaq, "history", _history_from(NASDAQ["history"]))
    monkeypatch.setattr(yahoo, "screener_gainers", lambda *a, **k: [dict(r) for r in YAHOO["rows"]])
    monkeypatch.setattr(nasdaq, "screener_gainers", lambda: [dict(r) for r in NASDAQ["screener_rows"]])

    def nasdaq_only(rows=None, history=None):
        monkeypatch.setattr(yahoo, "screener_gainers", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yahoo down")))
        if rows is not None:
            monkeypatch.setattr(nasdaq, "screener_gainers", lambda: [dict(r) for r in rows])
        if history is not None:
            monkeypatch.setattr(nasdaq, "history", _history_from(history))
    return nasdaq_only


def _screen() -> dict:
    return _payload(gainers.get_top_nasdaq_gainers.func(state={}, tool_call_id="screen"))


# --- quote freshness -------------------------------------------------------------

def test_freshness_needs_price_and_move_to_match_one_session():
    closes = {D21: 8.81, D22: 24.83, D23: 17.71}
    assert quote_freshness(17.71, -28.68, closes, D23, D22) == "fresh"
    assert quote_freshness(24.83, 181.84, closes, D23, D22) == "stale"       # TANH as Nasdaq served it
    assert quote_freshness(17.71, 181.84, closes, D23, D22) == "unknown"     # price and move disagree
    assert quote_freshness(17.71, -28.68, {}, D23, D22) == "unknown"


def test_equal_consecutive_closes_are_decided_by_the_move_not_the_price():
    closes = {D21: 1.0, D22: 2.0, D23: 2.0}
    assert quote_freshness(2.0, 0.0, closes, D23, D22) == "fresh"
    assert quote_freshness(2.0, 100.0, closes, D23, D22) == "stale"
    assert quote_freshness(2.0, 0.0, {D21: 2.0, D22: 2.0, D23: 2.0}, D23, D22) == "unknown"   # fits both days


# --- Yahoo: per-row dates ----------------------------------------------------------

def test_yahoo_rows_from_earlier_sessions_leave_the_ranking(ctx, market):
    stale = {r["symbol"] for r in YAHOO["rows"] if r["market_time"] != "2026-10-23"}
    assert {"ATII", "ADNWW"} <= stale          # ATII was quoted on the 21st, ADNWW on the 22nd

    payload = _screen()

    symbols = [c["symbol"] for c in payload["data"]["candidates"]]
    assert symbols[0] == "AIOK" and not stale & set(symbols)
    assert any(w.startswith(f"excluded {len(stale)} quotes") for w in payload["warnings"])
    # Every off-date row above AIOK is a warrant/right, so it cannot outrank a common stock.
    assert payload["data"]["ranking"]["status"] == "verified"


def test_an_off_date_common_stock_above_the_pick_leaves_the_ranking_unconfirmed(ctx, market, monkeypatch):
    rows = [dict(r, pct_change=250.0) if r["symbol"] == "ATII" else dict(r) for r in YAHOO["rows"]]
    monkeypatch.setattr(yahoo, "screener_gainers", lambda *a, **k: rows)

    payload = _screen()

    ranking = payload["data"]["ranking"]
    assert payload["data"]["candidates"][0]["symbol"] == "AIOK"
    assert ranking["status"] == "uncertain" and ranking["unresolved"][0]["symbol"] == "ATII"
    assert "ranking_unverified" in payload["data"]["candidates"][0]["flags"]
    assert "not a confirmed #1" in ranking["message"]


# --- Nasdaq: per-row history checks --------------------------------------------------

def test_lagging_nasdaq_screener_never_yields_a_confirmed_ranking(ctx, market):
    market()   # the recorded Nasdaq screener, captured while it still served 22 October

    payload = _screen()

    data = payload["data"]
    assert payload["ok"] and data["source"] == "nasdaq"
    assert {"TANH", "XLAB"} <= set(data["ranking"]["ruled_out"])      # proven stale; their real 23 Oct moves are lower
    assert data["ranking"]["status"] == "uncertain"
    assert {"unverified_quote", "ranking_unverified"} <= set(data["candidates"][0]["flags"])
    assert data["ranking"]["message"].startswith("no candidate could be verified")


def test_nasdaq_source_is_rejected_when_every_checked_row_is_stale(ctx, market):
    stale_rows = [r for r in NASDAQ["screener_rows"] if r["symbol"] in {"TANH", "XLAB"}]
    market(rows=stale_rows)

    payload = _screen()

    assert payload["ok"] is False and "every checked screener row belongs to a session before" in payload["error"]["message"]


def _nasdaq_rows(*rows):
    return [{"symbol": s, "name": f"{s} Common Stock", "price": p, "pct_change": pct, "volume": v} for s, p, pct, v in rows]


def test_nasdaq_rows_are_resolved_one_by_one(ctx, market):
    history = {"FLAT": _closes(1.0, 2.0, 2.0), "STALEUP": _closes(2.72, 4.71, 3.35), "FRESH": _closes(2.0, 2.0, 3.0)}
    rows = _nasdaq_rows(("FLAT", 2.0, 100.0, 900_000),       # previous session; same close today
                        ("STALEUP", 4.71, 73.16, 900_000),  # previous session; fell today
                        ("NOHIST", 5.0, 60.0, 900_000),     # cannot be checked
                        ("FRESH", 3.0, 50.0, 900_000))
    market(rows=rows, history=history)

    data = _screen()["data"]

    assert [c["symbol"] for c in data["candidates"]] == ["NOHIST", "FRESH"]
    assert data["selection"]["primary_symbol"] == "FRESH"
    assert data["ranking"]["status"] == "uncertain"
    assert [u["symbol"] for u in data["ranking"]["unresolved"]] == ["NOHIST"]
    assert {"FLAT", "STALEUP"} <= set(data["ranking"]["ruled_out"])


def test_nasdaq_ranking_is_verified_when_every_row_above_is_resolved(ctx, market):
    history = {"STALEUP": _closes(2.72, 4.71, 3.35), "FRESH": _closes(2.0, 2.0, 3.0)}
    market(rows=_nasdaq_rows(("STALEUP", 4.71, 73.16, 900_000), ("FRESH", 3.0, 50.0, 900_000)), history=history)

    assert _screen()["data"]["ranking"]["status"] == "verified"


def test_intraday_nasdaq_quotes_cannot_be_called_a_confirmed_first(ctx, market, monkeypatch):
    monkeypatch.setattr(gainers, "market_session", lambda: MarketSession(D23, True, D22))
    market(rows=_nasdaq_rows(("FRESH", 3.0, 50.0, 900_000)))

    data = _screen()["data"]

    assert data["ranking"]["status"] == "uncertain" and "unverified_quote" in data["candidates"][0]["flags"]


# --- filter boundaries are decided before elimination -----------------------------------

BOUNDARY = {"symbol": "EDGE", "name": "Edge Corp Common Stock", "price": 3.0, "pct_change": 300.0, "net_change": 2.25,
            "volume": 95_000, "market_cap": 60_000_000, "exchange": "NCM", "quote_type": "EQUITY", "market_time": "2026-10-23"}


@pytest.mark.parametrize("quote, daily, expected", [
    ({"volume": 95_000}, {"volume": 105_000}, "eligibility_unconfirmed"),   # screener below, daily above
    ({"volume": 105_000}, {"volume": 95_000}, "eligibility_unconfirmed"),   # screener above, daily below
    ({"price": 0.95}, {"close": 1.02}, "eligibility_unconfirmed"),          # price near $1
    ({"volume": 95_000}, {"volume": 90_000}, None),                         # both below: ruled out
    ({"volume": 105_000}, {"volume": 104_000}, "near_threshold"),           # both above: eligible
])
def test_rows_near_a_filter_are_checked_on_the_daily_bar(ctx, market, monkeypatch, quote, daily, expected):
    row = {**BOUNDARY, **quote}
    monkeypatch.setattr(yahoo, "screener_gainers", lambda *a, **k: [row, *(dict(r) for r in YAHOO["rows"])])
    bar = {"close": row["price"], "volume": 1_000_000, **daily}
    monkeypatch.setattr(yahoo, "history", lambda symbol, *a: (_frame(_closes(1.0, 0.75, bar["close"], volume=bar["volume"])), {}, {}))

    data = _screen()["data"]

    edge = next((c for c in data["candidates"] if c["symbol"] == "EDGE"), None)
    if expected is None:
        assert edge is None and "EDGE" in data["ranking"]["ruled_out"] and data["ranking"]["status"] == "verified"
    elif expected == "near_threshold":
        assert edge["flags"] == ["near_threshold"] and data["selection"]["primary_symbol"] == "EDGE"
    else:
        assert expected in edge["flags"]
        assert data["selection"]["primary_symbol"] == "AIOK"                 # kept, but not declared eligible
        assert data["ranking"]["status"] == "uncertain"


# --- the selected stock: price agreement and eligibility ---------------------------------

def test_aiok_price_is_consistent_across_two_vendors_and_volumes_are_recorded():
    mine = prices.normalise(pd.read_csv(LIVE / "yahoo_AIOK_history.csv"))
    theirs = prices.normalise(_frame(NASDAQ["history"]["AIOK"]))

    result = price_confirmation(mine, theirs, D23, primary_source="yahoo", secondary_source="nasdaq", conflict_pp=5.0)

    assert result.status == "consistent" and not result.flags and "consistent across" in result.message
    assert result.volume == {"yahoo": 42_564_200, "nasdaq": 42_891_980}
    observations = {"screener": (12.02, 41_847_594), "yahoo daily": (12.02, 42_564_200), "nasdaq daily": (12.02, 42_891_980)}
    assert eligibility_check(observations, {"min_price": 1, "min_volume": 100_000})["status"] == "confirmed"


def test_eligibility_is_unconfirmed_when_any_source_fails_the_filter():
    observations = {"screener": (3.0, 105_000), "yahoo daily": (3.0, 104_000), "nasdaq daily": (3.0, 96_000)}
    result = eligibility_check(observations, {"min_volume": 100_000})
    assert result["status"] == "unconfirmed" and result["failing"] == ["nasdaq daily"]
    assert eligibility_check(observations, {})["status"] == "not_applied"


def test_vendor_price_disagreement_marks_the_ranking_unverified(ctx, offline_market, monkeypatch):
    monkeypatch.setattr(nasdaq, "history", lambda *a, **k: bars(STOCK[:-1] + [STOCK[-1] * 1.03]))

    command = get_price_history.func(symbol="AIOK", state={"candidates": SCREENER, "session": SESSION.as_dict()},
                                     tool_call_id="h")

    assert "source_conflict" in command.update["selected"]["flags"]
    assert command.update["checks"]["price_confirmation"]["status"] == "conflict"
    limits = evidence_report._limits({"selected": command.update["selected"]}, {"benchmark": {"GSPC": 0, "IXIC": 0}})
    assert any("vendors disagree" in line for line in limits)


def test_missing_second_vendor_is_disclosed_not_hidden(ctx, offline_market, monkeypatch):
    monkeypatch.setattr(nasdaq, "history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nasdaq down")))

    command = get_price_history.func(symbol="AIOK", state={"candidates": SCREENER, "session": SESSION.as_dict()},
                                     tool_call_id="h")

    assert command.update["checks"]["price_confirmation"]["status"] == "single_source"
    assert any("second-vendor price check unavailable" in w for w in _payload(command)["warnings"])


def test_second_vendor_below_the_volume_filter_leaves_eligibility_unconfirmed(ctx, offline_market, monkeypatch):
    monkeypatch.setattr(nasdaq, "history", lambda *a, **k: bars(STOCK).assign(Volume=900_000))
    state = {"candidates": SCREENER, "session": SESSION.as_dict(), "selection": {"filters": {"min_volume": 1_000_000}}}

    command = get_price_history.func(symbol="AIOK", state=state, tool_call_id="h")

    assert "eligibility_unconfirmed" in command.update["selected"]["flags"]
    assert command.update["checks"]["filter_eligibility"]["failing"] == ["nasdaq daily"]


def test_ranking_limitation_names_the_unresolved_rows():
    state = {"selected": {"flags": ["ranking_unverified"]},
             "ranking": {"unresolved": [{"symbol": "ATII", "reason": "quote dated 2026-10-21"}]}}
    limits = evidence_report._limits(state, {"benchmark": {"GSPC": 0, "IXIC": 0}})
    assert any("(ATII)" in line and "not a confirmed #1" in line for line in limits)

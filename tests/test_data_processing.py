from datetime import date, datetime, timezone

import pandas as pd
import pytest

from nasdaq_agent.data import corporate_actions, cross_check, prices
from nasdaq_agent.data.calendar import market_session
from nasdaq_agent.data.screener_clean import apply_filters, clean_rows, is_derivative, parse_number
from nasdaq_agent.infra.errors import ToolError


@pytest.mark.parametrize("raw, expected", [
    ("$24.83", 24.83), ("181.839%", 181.839), ("2,212,652", 2212652), ("+7.77", 7.77), ("NA", None), ("", None), (3, 3.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("symbol, name, expected", [
    ("XLABW", "XLab Inc Warrant", True),
    ("HUBCZ", "Hub Cyber Security Ltd", True),    # 5th letter Z
    ("ABCDR", "Something Rights", True),
    ("AIOK", "AI OKTO CORP. Common Stock", False),
    ("TANH", "Tantech Holdings Ltd. Class A Common Shares", False),
])
def test_derivative_detection(symbol, name, expected):
    assert is_derivative(symbol, name) is expected


def test_clean_rows_flags_and_sorting():
    rows, dropped = clean_rows([
        {"symbol": "AIOK", "name": "AI OKTO", "price": 12.02, "pct_change": 182.82, "net_change": 7.77, "volume": 4e7, "market_cap": 5e7},
        {"symbol": "XLABW", "name": "XLab Warrant", "price": 0.38, "pct_change": 300.0, "volume": 500},
        {"symbol": "BAD", "name": "no price", "price": None, "pct_change": 50},
        {"symbol": "ODD", "name": "Odd Corp", "price": 10.0, "pct_change": 100.0, "net_change": 1.0, "volume": 2e6},
    ], "test")
    assert dropped == 1
    assert [r.symbol for r in rows] == ["XLABW", "AIOK", "ODD"]
    assert {"derivative", "penny", "low_volume"} <= set(rows[0].flags)
    assert rows[1].flags == []
    assert "inconsistent" in rows[2].flags  # +100% implies prev close 5.0, but price - change = 9.0
    filtered = apply_filters(rows, exclude_derivatives=True, min_price=1, min_volume=100_000, min_market_cap=None)
    assert [r.symbol for r in filtered] == ["AIOK", "ODD"]


def _frame(closes, end=date(2026, 10, 23)):
    dates = pd.bdate_range(end=end, periods=len(closes)).date
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1})


def test_validate_window_trims_and_checks():
    window, check = prices.validate_window(_frame([1, 2, 3, 4, 5, 6, 7, 8]), symbol="X", days=5,
                                           trade_date=date(2026, 10, 23), is_intraday=False, outlier_move=0.8)
    assert len(window) == 6 and check.last_date == "2026-10-23"


def test_validate_window_insufficient_history():
    with pytest.raises(ToolError) as exc:
        prices.validate_window(_frame([1, 2, 3]), symbol="NEW", days=5, trade_date=date(2026, 10, 23),
                               is_intraday=False, outlier_move=0.8)
    assert exc.value.type == "insufficient_history" and "days=2" in exc.value.hint


def test_validate_window_stale_history():
    with pytest.raises(ToolError) as exc:
        prices.validate_window(_frame([1, 2, 3, 4, 5, 6], end=date(2026, 10, 22)), symbol="X", days=5,
                               trade_date=date(2026, 10, 23), is_intraday=False, outlier_move=0.8)
    assert exc.value.type == "stale_history"


def test_outlier_flag():
    _, check = prices.validate_window(_frame([1, 1, 5, 5, 5, 5]), symbol="X", days=5,
                                      trade_date=date(2026, 10, 23), is_intraday=False, outlier_move=0.8)
    assert "outlier" in check.flags


def test_align_keeps_shared_dates():
    a, b = _frame([1, 2, 3, 4]), _frame([1, 2, 3])
    aligned, warnings = prices.align({"A": a, "B": b})
    assert len(aligned["A"]) == len(aligned["B"]) == 3 and warnings


def test_cross_check_levels():
    w = _frame([10, 20])
    assert cross_check.compare_day_move(w, 100.0, 1, 5).status == "ok"
    assert cross_check.compare_day_move(w, 97.0, 1, 5).status == "warn"
    res = cross_check.compare_day_move(w, 150.0, 1, 5)
    assert res.status == "conflict" and "data_conflict" in res.flags


def test_stale_quote_detection():
    closes = {date(2026, 10, 21): 8.81, date(2026, 10, 22): 24.83, date(2026, 10, 23): 17.71}
    assert cross_check.quote_freshness(24.83, 181.84, closes, date(2026, 10, 23), date(2026, 10, 22)) == "stale"
    assert cross_check.quote_freshness(17.71, -28.68, closes, date(2026, 10, 23), date(2026, 10, 22)) == "fresh"
    assert cross_check.quote_freshness(17.71, -28.68, {}, date(2026, 10, 23), date(2026, 10, 22)) == "unknown"


def test_reverse_split_artifact():
    actions = corporate_actions.from_events({date(2026, 10, 23): 0.1}, {})
    assert actions[0].kind == "reverse_split"
    assert corporate_actions.possible_split_artifact(actions, date(2026, 10, 23), 880.0)
    assert not corporate_actions.possible_split_artifact(actions, date(2026, 10, 23), 40.0)


def _split_bars(opens, closes):
    days = [date(2026, 10, d) for d in (20, 21, 22, 23, 26)][-len(closes):]
    return pd.DataFrame({"date": days, "open": opens, "high": [max(o, c) for o, c in zip(opens, closes)],
                         "low": [min(o, c) for o, c in zip(opens, closes)], "close": closes,
                         "volume": [100_000] * (len(closes) - 1) + [3_000_000]})


def test_unadjusted_bars_are_restated_when_they_jump_by_the_split_factor():
    # 1-for-2 reverse split on Oct 26: the open doubles the previous close, so the earlier bars are pre-split.
    frame = _split_bars([1.30, 1.25, 0.80, 1.62], [1.28, 1.20, 0.806, 1.61])
    actions = corporate_actions.from_events({date(2026, 10, 26): 0.5}, {})
    adjusted, [check] = corporate_actions.adjust_for_splits(frame, actions)
    assert (check.status, check.open, check.previous_close) == ("adjusted", 1.62, 0.806)
    assert list(adjusted["close"]) == [2.56, 2.40, 1.612, 1.61]
    assert list(adjusted["volume"]) == [50_000, 50_000, 50_000, 3_000_000]
    assert round(adjusted["close"].iloc[-1] / adjusted["close"].iloc[-2] - 1, 4) == -0.0012


def test_bars_without_a_split_sized_gap_are_left_as_they_are():
    # GCTK, 26 Oct 2026: Yahoo dates the split that day, yet the session opened at the pre-split level.
    frame = _split_bars([1.30, 1.25, 0.84, 0.84], [1.28, 1.20, 0.806, 1.61])
    actions = corporate_actions.from_events({date(2026, 10, 26): 0.5}, {})
    adjusted, [check] = corporate_actions.adjust_for_splits(frame, actions)
    assert check.status == "no_gap" and adjusted.equals(frame)
    # A vendor that already adjusted shows no gap either.
    already = _split_bars([2.60, 2.50, 1.68, 1.66], [2.56, 2.40, 1.612, 1.61])
    assert corporate_actions.adjust_for_splits(already, actions)[1][0].status == "no_gap"


def test_forward_split_inside_the_window_is_restated_and_bars_outside_are_ignored():
    frame = _split_bars([40.0, 41.0, 20.6, 20.8], [40.5, 41.2, 20.7, 21.0])
    actions = corporate_actions.from_events({date(2026, 10, 23): 2.0, date(2026, 9, 1): 0.5}, {})
    adjusted, checks = corporate_actions.adjust_for_splits(frame, actions)
    assert [c.status for c in checks] == ["adjusted"]
    assert list(adjusted["close"]) == [20.25, 20.6, 20.7, 21.0]
    assert list(adjusted["volume"][:2]) == [200_000, 200_000]


@pytest.mark.parametrize("now, expected_date, intraday", [
    (datetime(2026, 10, 24, 15, 0, tzinfo=timezone.utc), date(2026, 10, 23), False),   # Saturday
    (datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc), date(2026, 10, 23), False),   # Monday pre-market
    (datetime(2026, 10, 26, 15, 0, tzinfo=timezone.utc), date(2026, 10, 26), True),    # Monday 11:00 ET
    (datetime(2026, 10, 26, 21, 0, tzinfo=timezone.utc), date(2026, 10, 26), False),   # Monday after close
    (datetime(2026, 11, 26, 18, 0, tzinfo=timezone.utc), date(2026, 11, 25), False),   # Thanksgiving
])
def test_market_session(now, expected_date, intraday):
    s = market_session(now)
    assert s.trade_date == expected_date and s.is_intraday is intraday

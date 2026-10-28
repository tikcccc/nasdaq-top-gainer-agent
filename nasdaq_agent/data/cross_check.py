"""Compare independent sources so bad or stale data is caught before analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CrossCheck:
    screener_pct: float
    history_pct: float | None
    gap_pp: float | None
    status: str                 # "ok" | "warn" | "conflict" | "unverified"
    message: str
    flags: list[str] = field(default_factory=list)


def compare_day_move(window: pd.DataFrame, screener_pct: float, warn_pp: float, conflict_pp: float) -> CrossCheck:
    """Screener % change vs the % change implied by the last two closes of the history."""
    if len(window) < 2:
        return CrossCheck(screener_pct, None, None, "unverified", "not enough bars to verify the day move")
    history_pct = float((window["close"].iloc[-1] / window["close"].iloc[-2] - 1) * 100)
    gap = abs(history_pct - screener_pct)
    if gap >= conflict_pp:
        return CrossCheck(
            screener_pct, history_pct, gap, "conflict",
            f"screener says {screener_pct:+.2f}% but price history implies {history_pct:+.2f}% ({gap:.1f}pp apart)",
            ["data_conflict"],
        )
    if gap >= warn_pp:
        return CrossCheck(
            screener_pct, history_pct, gap, "warn",
            f"screener {screener_pct:+.2f}% vs history {history_pct:+.2f}% ({gap:.1f}pp apart)",
        )
    return CrossCheck(screener_pct, history_pct, gap, "ok", f"screener and history agree ({gap:.2f}pp)")


def day_move(closes_by_date: dict, day) -> float | None:
    """% change of `day` against the preceding bar in the series, or None when either is missing."""
    earlier = [d for d in closes_by_date if d < day]
    if day not in closes_by_date or not earlier or not closes_by_date[max(earlier)]:
        return None
    return (closes_by_date[day] / closes_by_date[max(earlier)] - 1) * 100


def quote_freshness(price: float, pct_change: float, closes_by_date: dict, trade_date, previous_date,
                    *, pct_tol_pp: float = 1.0) -> str:
    """Which session an untimestamped screener quote belongs to.

    'fresh' when price and % change both match the trade-date bar, 'stale' when both match
    the previous session instead, 'unknown' otherwise. A price alone is not enough: a stock
    can close at the same price two days running."""
    def fits(day) -> bool:
        close, move = closes_by_date.get(day), day_move(closes_by_date, day)
        return bool(close) and move is not None and abs(price / close - 1) < 0.005 and abs(pct_change - move) <= pct_tol_pp

    today, previous = fits(trade_date), fits(previous_date)
    return "fresh" if today and not previous else "stale" if previous and not today else "unknown"


@dataclass
class Confirmation:
    status: str                 # "consistent" | "single_source" | "conflict"
    primary: str
    secondary: str | None
    message: str
    close: dict = field(default_factory=dict)
    day_move_pct: dict = field(default_factory=dict)
    volume: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _day(frame: pd.DataFrame, trade_date) -> tuple[float, float, float | None] | None:
    """(close, day move %, volume) on the trade date, or None when either session is missing."""
    rows = frame[frame["date"] <= trade_date].tail(2)
    if len(rows) < 2 or rows["date"].iloc[-1] != trade_date:
        return None
    close, prev = float(rows["close"].iloc[-1]), float(rows["close"].iloc[-2])
    volume = rows["volume"].iloc[-1]
    return close, (close / prev - 1) * 100, None if pd.isna(volume) else float(volume)


def price_confirmation(primary: pd.DataFrame, secondary: pd.DataFrame | None, trade_date, *,
                               primary_source: str, secondary_source: str | None,
                               conflict_pp: float, price_tol: float = 0.005) -> Confirmation:
    """Check the selected stock's trade-date close and day move against a second vendor.

    This supports the price, not the ranking: other stocks may still have risen more.
    The vendors may share upstream feeds, so agreement means "consistent", not
    "independently proven". Volume is recorded but never decides the status."""
    mine = _day(primary, trade_date)
    theirs = _day(secondary, trade_date) if secondary is not None else None
    if mine is None or theirs is None:
        reason = "no second vendor responded" if secondary is None else f"{secondary_source} lacks the trade-date bars"
        return Confirmation("single_source", primary_source, secondary_source,
                            f"price history from {primary_source} only; second-vendor price check unavailable ({reason})")
    names = (primary_source, secondary_source)
    result = Confirmation("consistent", primary_source, secondary_source, "",
                          close=dict(zip(names, (mine[0], theirs[0]))),
                          day_move_pct=dict(zip(names, (round(mine[1], 4), round(theirs[1], 4)))),
                          volume=dict(zip(names, (mine[2], theirs[2]))))
    close_gap = abs(mine[0] / theirs[0] - 1)
    move_gap = abs(mine[1] - theirs[1])
    if close_gap > price_tol or move_gap >= conflict_pp:
        result.status, result.flags = "conflict", ["source_conflict"]
        result.message = (f"{primary_source} and {secondary_source} disagree on the trade date: close "
                          f"{mine[0]:.4g} vs {theirs[0]:.4g}, day move {mine[1]:+.2f}% vs {theirs[1]:+.2f}%")
    else:
        result.message = f"close and day move consistent across {primary_source} and {secondary_source}"
    return result

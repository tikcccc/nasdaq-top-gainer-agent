"""Normalise and validate daily price history before the analysis step sees it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import numpy as np

from nasdaq_agent.data.calendar import _cal
from nasdaq_agent.infra.errors import ToolError

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class PriceCheck:
    rows: int
    first_date: str
    last_date: str
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    processing: list[str] = field(default_factory=list)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Unify column names, index and types. Input may come from yfinance or Nasdaq."""
    frame = df.copy()
    if not isinstance(frame.index, pd.RangeIndex) or "date" not in {str(c).lower() for c in frame.columns}:
        frame = frame.reset_index()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    frame = frame.rename(columns={"index": "date", "datetime": "date"})
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ToolError("bad_schema", f"price data missing columns {missing}", retryable=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=False).map(
        lambda ts: ts.tz_convert("America/New_York").date() if getattr(ts, "tzinfo", None) else ts.date()
    )
    for col in COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[COLUMNS].drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    return frame


def validate_window(
    frame: pd.DataFrame,
    *,
    symbol: str,
    days: int,
    trade_date: date,
    is_intraday: bool,
    outlier_move: float,
) -> tuple[pd.DataFrame, PriceCheck]:
    """Trim to ``days + 1`` sessions ending at ``trade_date`` and run sanity checks."""
    processing = ["columns normalised; split-adjusted closes (not dividend-adjusted)"]
    frame = frame[frame["date"] <= trade_date]
    before = len(frame)
    frame = frame.dropna(subset=["close"])
    if len(frame) < before:
        processing.append(f"dropped {before - len(frame)} rows with missing close")
    if (frame["close"] <= 0).any() or not np.isfinite(frame["close"]).all():
        raise ToolError("bad_data", f"{symbol}: non-positive close prices", retryable=True, hint="Retry; another source will be used.")

    need = days + 1
    window = frame.tail(need).reset_index(drop=True)
    warnings: list[str] = []
    flags: list[str] = []
    if len(window) < need:
        raise ToolError(
            "insufficient_history",
            f"{symbol} has only {len(window)} sessions up to {trade_date} (need {need} for {days} daily returns).",
            retryable=False,
            hint=(
                f"Possibly a recent IPO or trading halt (available_days={max(len(window) - 1, 0)}). "
                "Report the first-ranked stock's limitation; changing the requested window requires new user settings."
            ),
        )
    if window["date"].iloc[-1] != trade_date:
        if is_intraday:
            warnings.append(f"{symbol}: no bar for {trade_date} yet (intraday run); latest is {window['date'].iloc[-1]}")
        else:
            raise ToolError(
                "stale_history",
                f"{symbol}: latest bar is {window['date'].iloc[-1]}, expected {trade_date}.",
                retryable=True,
                hint="Retry once (a fallback source will be tried); if it persists choose another candidate.",
            )
    expected_end = window["date"].iloc[-1] if is_intraday else trade_date
    expected = [d.date() for d in _cal().sessions_window(pd.Timestamp(expected_end), -(days + 1))]
    if list(window["date"]) != expected:
        raise ToolError("missing_sessions", f"{symbol}: window is not the last {days + 1} consecutive sessions",
                        retryable=True, hint="Try a source with complete trading-session history.")
    for col in ("open", "high", "low", "volume"):
        if not np.isfinite(window[col]).all() or (window[col] < (0 if col == "volume" else 1e-300)).any():
            raise ToolError("bad_data", f"{symbol}: invalid {col}", retryable=True)
    if ((window.high < window[["open", "close", "low"]].max(axis=1)) |
            (window.low > window[["open", "close", "high"]].min(axis=1))).any():
        raise ToolError("bad_data", f"{symbol}: inconsistent OHLC bars", retryable=True)
    returns = window["close"].pct_change().iloc[1:]
    for d, r in zip(window["date"].iloc[1:-1], returns.iloc[:-1]):
        if abs(r) > outlier_move:
            flags.append("outlier")
            warnings.append(f"{symbol}: {d} moved {r:+.0%} (outside the trade date) - check corporate actions")
    return window, PriceCheck(
        rows=len(window),
        first_date=str(window["date"].iloc[0]),
        last_date=str(window["date"].iloc[-1]),
        warnings=warnings,
        flags=sorted(set(flags)),
        processing=processing,
    )


def align(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Keep only dates shared by every series (inner join on date)."""
    if len(frames) < 2:
        return frames, []
    common = set.intersection(*(set(f["date"]) for f in frames.values()))
    warnings = []
    aligned = {}
    for key, frame in frames.items():
        dropped = sorted(set(frame["date"]) - common)
        if dropped:
            warnings.append(f"{key}: dropped {len(dropped)} non-shared dates when aligning ({dropped[:3]})")
        aligned[key] = frame[frame["date"].isin(common)].reset_index(drop=True)
    return aligned, warnings

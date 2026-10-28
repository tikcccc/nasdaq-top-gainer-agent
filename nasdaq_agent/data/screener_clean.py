"""Parse, flag and filter top-gainer rows from any screener source."""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import ValidationError

from nasdaq_agent.infra.schemas import (
    FLAG_DERIVATIVE,
    FLAG_INCONSISTENT,
    FLAG_LOW_VOLUME,
    FLAG_MICRO_CAP,
    FLAG_PENNY,
    GainerRow,
)

_DERIVATIVE_NAME = re.compile(
    r"\b(warrants?|rights?|units?|preferred|depositary\s+shares?\s+each\s+representing|notes?\s+due|debentures?)\b",
    re.IGNORECASE,
)
# NASDAQ 5th-letter suffixes: W = warrant, R = rights, U = units, Z = misc.
_DERIVATIVE_SUFFIX = re.compile(r"^[A-Z]{4}[WRUZ]$|[./^-](W|WS|WT|R|RT|U|UN)$")

LOW_VOLUME = 100_000
MICRO_CAP = 50_000_000


def parse_number(value: Any) -> float | None:
    """'$24.83' -> 24.83, '181.839%' -> 181.839, '2,212,652' -> 2212652, 'NA'/'' -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    if text in {"", "NA", "N/A", "--", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_derivative(symbol: str, name: str) -> bool:
    return bool(_DERIVATIVE_NAME.search(name or "")) or bool(_DERIVATIVE_SUFFIX.search(symbol.upper()))


def compute_flags(row: GainerRow, *, low_volume: int = LOW_VOLUME, micro_cap: float = MICRO_CAP) -> list[str]:
    flags: list[str] = []
    if is_derivative(row.symbol, row.name) or (row.quote_type and row.quote_type.upper() not in {"EQUITY", ""}):
        flags.append(FLAG_DERIVATIVE)
    if row.price < 1:
        flags.append(FLAG_PENNY)
    if row.volume is not None and row.volume < low_volume:
        flags.append(FLAG_LOW_VOLUME)
    if row.market_cap is not None and 0 < row.market_cap < micro_cap:
        flags.append(FLAG_MICRO_CAP)
    if row.net_change is not None and row.pct_change > -100:
        # previous close implied by pct must match price - net change
        prev_from_pct = row.price / (1 + row.pct_change / 100)
        prev_from_change = row.price - row.net_change
        if prev_from_change > 0 and abs(prev_from_pct / prev_from_change - 1) > 0.01:
            flags.append(FLAG_INCONSISTENT)
    return flags


def clean_rows(raw: Iterable[dict[str, Any]], source: str) -> tuple[list[GainerRow], int]:
    """Validate rows into GainerRow, attach flags, sort by % change. Returns (rows, dropped)."""
    rows: list[GainerRow] = []
    dropped = 0
    seen: set[str] = set()
    for item in raw:
        try:
            row = GainerRow.model_validate({**item, "source": source})
        except ValidationError:
            dropped += 1
            continue
        if row.symbol in seen:
            continue
        seen.add(row.symbol)
        row.flags = compute_flags(row)
        rows.append(row)
    rows.sort(key=lambda r: r.pct_change, reverse=True)
    return rows, dropped


def split_by_quote_date(raw: Iterable[dict[str, Any]], trade_date: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate rows quoted on the trade date from rows carrying an older (or no) quote date.

    A screener sorted by % change mixes in symbols that have not traded since an earlier
    session; their change belongs to that session and must not enter today's ranking."""
    fresh, stale = [], []
    for item in raw:
        (fresh if item.get("market_time") == trade_date else stale).append(item)
    return fresh, stale


def apply_filters(
    rows: list[GainerRow],
    *,
    exclude_derivatives: bool,
    min_price: float | None,
    min_volume: int | None,
    min_market_cap: float | None,
) -> list[GainerRow]:
    out = []
    for row in rows:
        if exclude_derivatives and FLAG_DERIVATIVE in row.flags:
            continue
        if min_price is not None and row.price < min_price:
            continue
        if min_volume is not None and (row.volume or 0) < min_volume:
            continue
        if min_market_cap is not None and (row.market_cap or 0) < min_market_cap:
            continue
        out.append(row)
    return out

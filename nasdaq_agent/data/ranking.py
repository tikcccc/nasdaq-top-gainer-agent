"""Is the selected stock really the top gainer within the user's scope?

A second vendor agreeing on the selected stock's close says nothing about whether
another stock rose more. This module keeps the two questions apart: filter
eligibility near a threshold is decided from daily bars before anything is
eliminated, and the ranking is only "verified" when no row that could outrank the
selected stock is still unresolved.
"""

from __future__ import annotations

from typing import Any

PRICE_VOLUME = (("price", "min_price"), ("volume", "min_volume"))


def passes(price: float | None, volume: float | None, filters: dict) -> bool:
    if filters.get("min_price") and (price is None or price < filters["min_price"]):
        return False
    return not (filters.get("min_volume") and (volume is None or volume < filters["min_volume"]))


def relaxed(filters: dict, band: float) -> dict:
    """Thresholds lowered by `band`, so rows just below a limit survive until checked."""
    out = dict(filters)
    if out.get("min_price"):
        out["min_price"] = out["min_price"] * (1 - band)
    if out.get("min_volume"):
        out["min_volume"] = int(out["min_volume"] * (1 - band))
    return out


def near_thresholds(price: float, volume: float | None, filters: dict, band: float) -> list[str]:
    """Price/volume thresholds this quote sits within `band` of, on either side."""
    values = {"price": price, "volume": volume}
    return [key for field, key in PRICE_VOLUME
            if filters.get(key) and values[field] is not None and abs(values[field] / filters[key] - 1) < band]


def eligibility(price: float, volume: float | None, daily: dict | None, filters: dict) -> str:
    """'eligible' when the screener quote and the daily bar both pass, 'ineligible' when both fail,
    otherwise 'unconfirmed': the row is kept but not declared eligible."""
    if not daily:
        return "unconfirmed"
    quote_ok, daily_ok = passes(price, volume, filters), passes(daily.get("close"), daily.get("volume"), filters)
    return "eligible" if quote_ok and daily_ok else "ineligible" if not (quote_ok or daily_ok) else "unconfirmed"


def assess(primary: dict, contenders: list[dict[str, Any]], *, is_intraday: bool) -> dict:
    """Ranking status of `primary` given rows that ranked above it but are not verified candidates.

    A contender is ruled out only when its actual trade-date move is known and lower."""
    unresolved = [c for c in contenders
                  if c.get("actual_move") is None or c["actual_move"] > primary["pct_change"]]
    if primary.get("unresolved"):
        unresolved.insert(0, {"symbol": primary["symbol"], "reason": primary["unresolved"]})
    if unresolved:
        shown = ", ".join(f"{c['symbol']} ({c['reason']})" for c in unresolved[:4])
        lead = "no candidate could be verified" if primary.get("unresolved") else "top verified candidate, not a confirmed #1"
        status, message = "uncertain", f"{lead}: unresolved {shown}"
    elif is_intraday:
        status, message = "provisional_intraday", "highest verified intraday gain; changes until the close"
    else:
        status, message = "verified", "no unresolved row ranks above the selected stock"
    return {"status": status, "message": message,
            "unresolved": [{"symbol": c["symbol"], "reason": c["reason"]} for c in unresolved],
            "ruled_out": [c["symbol"] for c in contenders if c not in unresolved],
            # Kept so the ranking can be reassessed if the selected stock is later ruled out.
            "contenders": [{k: c.get(k) for k in ("symbol", "reason", "actual_move")} for c in contenders]}

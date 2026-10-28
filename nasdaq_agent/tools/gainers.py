"""get_top_nasdaq_gainers: today's biggest NASDAQ movers, cleaned, flagged and ranked with
an explicit statement of whether the #1 is verified."""

import time
from typing import Annotated

import pandas as pd
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from nasdaq_agent.context import get_ctx
from nasdaq_agent.data import prices, ranking
from nasdaq_agent.data.calendar import MarketSession, market_session, sessions_back
from nasdaq_agent.data.cross_check import day_move, quote_freshness
from nasdaq_agent.data.screener_clean import apply_filters, clean_rows, is_derivative, split_by_quote_date
from nasdaq_agent.infra.breaker import first_success
from nasdaq_agent.infra.cache import cached
from nasdaq_agent.infra.errors import StaleDataError, ToolError
from nasdaq_agent.infra.schemas import GainerRow
from nasdaq_agent.providers import nasdaq, yahoo
from nasdaq_agent.tools.base import ToolResult, agent_tool

FLAG_LEGEND = {
    "derivative": "warrant / right / unit / preferred - not a common stock",
    "penny": "price below $1",
    "low_volume": "fewer than 100k shares traded",
    "micro_cap": "market cap below $50M",
    "inconsistent": "price, change and % change do not reconcile",
    "near_threshold": "within the boundary band of a price/volume filter; eligibility checked on the daily bar",
    "eligibility_unconfirmed": "screener quote and daily bar disagree on the price/volume filter",
    "unverified_quote": "the quote could not be matched to this session's price history",
    "ranking_unverified": "a row that could rank higher is unresolved; this is not a confirmed #1",
}
UNRESOLVED = {"eligibility_unconfirmed": "filter eligibility unconfirmed", "unverified_quote": "quote not verified"}


class _Verifier:
    """Per-symbol daily bars for screener checks, bounded by a request count and a time budget."""

    def __init__(self, session: MarketSession):
        checks = get_ctx().settings.data_checks
        self.session = session
        self.left = checks.max_verification_requests
        self.deadline = time.monotonic() + checks.verification_budget_s
        self.ttl = 600 if session.is_intraday else 6 * 3600
        self.exhausted = False

    def closes(self, vendor: str, symbol: str) -> pd.DataFrame | None:
        """Recent normalised bars, or None when unavailable or the budget is spent."""
        if self.left <= 0 or time.monotonic() > self.deadline or get_ctx().seconds_left() < 60:
            self.exhausted = True
            return None
        self.left -= 1
        start, end = sessions_back(self.session.trade_date, 4), self.session.trade_date

        def fetch() -> list[dict]:
            bars = nasdaq.history(symbol, start, end) if vendor == "nasdaq" else yahoo.history(symbol, start, end)[0]
            frame = prices.normalise(bars)
            return frame.assign(date=frame["date"].astype(str)).to_dict("records")
        try:
            records, _ = cached(f"bars:{vendor}:{symbol}:{start}:{end}", self.ttl, fetch)
        except Exception:  # noqa: BLE001 - a missing history leaves the row unresolved
            return None
        frame = pd.DataFrame(records)
        return frame.assign(date=pd.to_datetime(frame["date"]).dt.date) if not frame.empty else None

    def daily(self, vendor: str, symbol: str) -> dict | None:
        frame = self.closes(vendor, symbol)
        bar = frame[frame["date"] == self.session.trade_date] if frame is not None else None
        if bar is None or bar.empty:
            return None
        return {"close": float(bar["close"].iloc[-1]), "volume": None if pd.isna(bar["volume"].iloc[-1]) else float(bar["volume"].iloc[-1])}


def _yahoo(session: MarketSession, warnings: list[str]) -> tuple[list[dict], list[dict]]:
    """Yahoo timestamps every quote: rows from an earlier session leave the ranking one by one."""
    ttl = 600 if session.is_intraday else 6 * 3600
    rows, _ = cached(f"screener:yahoo:{session.trade_date}:{session.is_intraday}", ttl,
                     lambda: yahoo.screener_gainers(session.trade_date, session.is_intraday))
    fresh, stale = split_by_quote_date(rows, session.trade_date.isoformat())
    if not fresh:
        raise StaleDataError("yahoo", f"no screener quote is dated {session.trade_date}")
    if stale:
        shown = ", ".join(f"{r.get('symbol')} ({r.get('market_time') or 'undated'})" for r in stale[:5])
        warnings.append(f"excluded {len(stale)} quote{'s' if len(stale) != 1 else ''} not dated {session.trade_date}: {shown}"
                        + (" ..." if len(stale) > 5 else ""))
    excluded = [{**r, "reason": f"quote dated {r.get('market_time') or 'unknown'}"} for r in stale]
    return fresh, excluded


def _nasdaq(session: MarketSession, warnings: list[str], filters: dict, limit: int,
            verifier: _Verifier) -> tuple[list[dict], list[dict]]:
    """Nasdaq's screener has no timestamp and can lag a session, so rows are checked against
    each symbol's own history in rank order. Proven-stale rows leave the ranking with their
    actual trade-date move recorded; unverifiable rows stay, flagged, and keep the ranking
    unconfirmed. Only when every checked row is stale is the source rejected."""
    ttl = 600 if session.is_intraday else 6 * 3600
    rows, _ = cached(f"screener:nasdaq:{session.trade_date}:{session.is_intraday}", ttl, nasdaq.screener_gainers)
    if session.is_intraday:
        warnings.append("nasdaq screener rows carry no timestamp; intraday candidates are unverified")
        return [{**r, "quote_check": "unknown"} for r in rows], []
    cleaned, _ = clean_rows(rows, "nasdaq")
    by_symbol = {r["symbol"].strip().upper(): r for r in rows if r.get("symbol")}
    band = get_ctx().settings.data_checks.filter_boundary_band
    tol = get_ctx().settings.data_checks.cross_source_warn_pp
    considered = apply_filters(cleaned, **ranking.relaxed(filters, band))
    out, excluded, fresh = [], [], 0
    for row in list({r.symbol: r for r in [*cleaned[:1], *considered]}.values()):
        if fresh >= limit or (verifier.exhausted and len(out) >= limit):
            break
        bars = verifier.closes("nasdaq", row.symbol)
        closes = dict(zip(bars["date"], bars["close"])) if bars is not None else {}
        verdict = quote_freshness(row.price, row.pct_change, closes, session.trade_date, session.previous_date,
                                  pct_tol_pp=tol)
        if verdict == "stale":
            excluded.append({**by_symbol[row.symbol], "symbol": row.symbol, "reason": "quote from the previous session",
                             "actual_move": day_move(closes, session.trade_date)})
            continue
        today = bars[bars["date"] == session.trade_date] if bars is not None else None
        daily = ({"close": float(today["close"].iloc[-1]), "volume": float(today["volume"].iloc[-1])}
                 if today is not None and not today.empty else None)
        out.append({**by_symbol[row.symbol], "symbol": row.symbol, "quote_check": verdict, "daily": daily,
                    "market_time": session.trade_date.isoformat() if verdict == "fresh" else None})
        fresh += verdict == "fresh"
    if not out:
        raise StaleDataError("nasdaq", f"every checked screener row belongs to a session before {session.trade_date}: "
                                       f"{[e['symbol'] for e in excluded[:5]]}")
    unknown = [r["symbol"] for r in out if r["quote_check"] == "unknown"]
    if excluded or unknown:
        warnings.append(f"nasdaq rows checked against history - stale: {[e['symbol'] for e in excluded[:5]]}, "
                        f"unverifiable: {unknown[:5]}" + ("; verification budget exhausted" if verifier.exhausted else ""))
    return out, excluded


def _select(rows: list[GainerRow], extra: dict, filters: dict, source: str, verifier: _Verifier) -> tuple[list[GainerRow], list[dict]]:
    """Apply the user's filters without eliminating rows near a threshold on the screener figure
    alone: those ranked above the first clearly eligible row are decided on the daily bar."""
    band = get_ctx().settings.data_checks.filter_boundary_band
    out: list[GainerRow] = []
    ruled_out: list[dict] = []
    settled = False
    for row in apply_filters(rows, **ranking.relaxed(filters, band)):
        info = extra.get(row.symbol, {})
        if info.get("quote_check") == "unknown":
            row.flags.append("unverified_quote")
        near = ranking.near_thresholds(row.price, row.volume, filters, band)
        if not near or settled:
            if ranking.passes(row.price, row.volume, filters):
                row.flags += ["near_threshold"] if near else []
                out.append(row)
                settled = settled or "unverified_quote" not in row.flags
            continue
        status = ranking.eligibility(row.price, row.volume, info.get("daily") or verifier.daily(source, row.symbol), filters)
        if status == "ineligible":
            ruled_out.append({"symbol": row.symbol, "reason": "fails the filter on the daily bar"})
            continue
        row.flags.append("near_threshold" if status == "eligible" else "eligibility_unconfirmed")
        out.append(row)
    return out, ruled_out


def _brief(row: GainerRow, rank: int) -> dict:
    return {
        "rank": rank,
        "symbol": row.symbol,
        "name": row.name[:60],
        "price": round(row.price, 4),
        "pct_change": round(row.pct_change, 2),
        "volume": row.volume,
        "market_cap": row.market_cap,
        "flags": row.flags,
    }


@tool("get_top_nasdaq_gainers")
@agent_tool
def get_top_nasdaq_gainers(
    limit: Annotated[int, Field(ge=1, le=50, description="How many candidates to return.")] = 10,
    exclude_derivatives: Annotated[bool | None, Field(description="Drop warrants, rights, units and preferreds (default: true).")] = None,
    min_price: Annotated[float | None, Field(ge=0, description="Minimum last price in USD (default from config; 0 disables).")] = None,
    min_volume: Annotated[int | None, Field(ge=0, description="Minimum shares traded on the day (default from config; 0 disables).")] = None,
    min_market_cap: Annotated[float | None, Field(ge=0, description="Minimum market cap in USD.")] = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolResult:
    """Get the NASDAQ stocks with the highest % gain on the most recent trading day.

    One call is enough: by default the candidates are already the top COMMON stocks (warrants,
    rights and units excluded; optional price/volume filters come from run settings), and the
    unfiltered #1 is always returned separately as raw_top. Filters are fixed by the user's run
    settings; omit filter arguments to use them. Returns the trade date the data belongs to,
    ranked candidates with quality flags and a ranking status: when it is not "verified" the
    first candidate is the top verified candidate, not a confirmed #1.
    Data comes from Yahoo Finance, falling back to Nasdaq.com."""
    settings = get_ctx().settings
    defaults = settings.screener.default_filters
    for name, supplied in (("exclude_derivatives", exclude_derivatives), ("min_price", min_price),
                           ("min_volume", min_volume), ("min_market_cap", min_market_cap)):
        if supplied is not None and supplied != getattr(defaults, name):
            raise ToolError("selection_scope_locked", f"{name} is fixed by the user's run settings",
                            hint="Omit filter arguments to use the user's selection scope. Do not change the ranking universe.")
    exclude_derivatives = defaults.exclude_derivatives if exclude_derivatives is None else exclude_derivatives
    min_price = defaults.min_price if min_price is None else (min_price or None)
    min_volume = defaults.min_volume if min_volume is None else (min_volume or None)
    min_market_cap = defaults.min_market_cap if min_market_cap is None else (min_market_cap or None)
    session = market_session()
    warnings: list[str] = []
    filters = {"exclude_derivatives": exclude_derivatives, "min_price": min_price,
               "min_volume": min_volume, "min_market_cap": min_market_cap}
    verifier = _Verifier(session)
    chain = first_success([
        ("yahoo", lambda: _yahoo(session, warnings)),
        ("nasdaq", lambda: _nasdaq(session, warnings, filters, limit, verifier)),
    ])
    warnings = chain.warnings + warnings
    raw_rows, excluded = chain.value
    extra = {str(r.get("symbol", "")).strip().upper(): r for r in raw_rows}
    rows, dropped = clean_rows(raw_rows, chain.source)
    if not rows:
        raise ToolError("no_data", "screener returned no usable rows", retryable=True)
    if dropped:
        warnings.append(f"dropped {dropped} malformed rows")
    if session.is_intraday:
        warnings.append("market is open: % changes are intraday and will move until 16:00 ET")

    filtered, ruled_out = _select(rows, extra, filters, chain.source, verifier)
    if not filtered:
        raise ToolError(
            "no_match", "no candidates pass the filters", retryable=False,
            hint="Report that no stock matches the user's filters; changing the scope requires new user settings.",
        )
    primary = next((r for r in filtered if not set(UNRESOLVED) & set(r.flags)), filtered[0])
    contenders = [{"symbol": r.symbol, "reason": UNRESOLVED[next(f for f in UNRESOLVED if f in r.flags)]}
                  for r in filtered[:filtered.index(primary)]]
    contenders += [{"symbol": e["symbol"], "reason": e["reason"], "actual_move": e.get("actual_move")}
                   for e in excluded if (e.get("pct_change") or 0) > primary.pct_change
                   and not (exclude_derivatives and is_derivative(str(e["symbol"]), e.get("name") or ""))]
    unresolved_self = next((UNRESOLVED[f] for f in UNRESOLVED if f in primary.flags), None)
    status = ranking.assess({"symbol": primary.symbol, "pct_change": primary.pct_change, "unresolved": unresolved_self},
                            contenders, is_intraday=session.is_intraday)
    status["ruled_out"] += [r["symbol"] for r in ruled_out]
    if status["status"] == "uncertain":
        primary.flags.append("ranking_unverified")
        warnings.append(f"ranking: {status['message']}")
    top = filtered[:limit] if primary in filtered[:limit] else [*filtered[:limit - 1], primary]
    raw_top = rows[0]
    data = {
        **session.as_dict(),
        "source": chain.source,
        "rows_scanned": len(rows),
        "filters": {k: v for k, v in filters.items() if v},
        "raw_top": _brief(raw_top, 1),
        "candidates": [_brief(r, i + 1) for i, r in enumerate(top)],
        "ranking": status,
        "flag_legend": {f: FLAG_LEGEND[f] for f in sorted({f for r in top + [raw_top] for f in r.flags}) if f in FLAG_LEGEND},
    }
    data["selection"] = {
        "filters": data["filters"], "primary_symbol": primary.symbol,
        "filtered": bool(min_price or min_volume or min_market_cap),
        "label": "Top filtered NASDAQ gainer" if min_price or min_volume or min_market_cap else "Top NASDAQ common-stock gainer",
    }
    return ToolResult(
        data=data,
        warnings=warnings,
        source=chain.source,
        update={
            "session": session.as_dict(),
            "candidates": [r.model_dump() for r in top],
            "raw_top": raw_top.model_dump(),
            "selection": data["selection"],
            "ranking": status,
            "selected": {**primary.model_dump(), "days": settings.analysis.days},
            "analysis": {},
            "sentiment": {},
            "datasets": {},
            "charts": [],
            "sources": {"screener": chain.source},
        },
    )

"""get_price_history: validated daily bars for the chosen stock and its benchmarks."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from nasdaq_agent.context import get_ctx
from nasdaq_agent.data import corporate_actions, prices, ranking
from nasdaq_agent.data.calendar import market_session, sessions_back
from nasdaq_agent.data.cross_check import compare_day_move, price_confirmation
from nasdaq_agent.infra.breaker import first_success
from nasdaq_agent.infra.errors import ToolError
from nasdaq_agent.providers import nasdaq, yahoo
from nasdaq_agent.tools.gainers import UNRESOLVED
from nasdaq_agent.tools.base import ToolResult, agent_tool


def dataset_id(symbol: str) -> str:
    return "ds_" + symbol.lstrip("^").upper().replace(".", "_")


def _stock_bars(symbol: str, start, end, *, days=5, is_intraday=False, outlier_move=0.8) -> tuple[pd.DataFrame, list | None, str, list[str]]:
    """Bars, corporate actions (None when no source could be checked), source and warnings."""
    def validated(bars):
        frame = prices.normalise(bars)
        prices.validate_window(frame, symbol=symbol, days=days, trade_date=end,
                               is_intraday=is_intraday, outlier_move=outlier_move)
        return frame

    yahoo_actions: list = []

    def from_yahoo():
        bars, splits, dividends = yahoo.history(symbol, start, end)
        # Kept even when the bars fail validation: a Nasdaq fallback has no split or dividend data.
        yahoo_actions.append(corporate_actions.from_events(splits, dividends))
        return validated(bars), yahoo_actions[0]

    def from_nasdaq():
        return validated(nasdaq.history(symbol, start, end, "stocks")), (yahoo_actions[0] if yahoo_actions else None)

    chain = first_success([("yahoo", from_yahoo), ("nasdaq", from_nasdaq)])
    frame, actions = chain.value
    return frame, actions, chain.source, chain.warnings


SECOND_SOURCE = {"yahoo": "nasdaq", "nasdaq": "yahoo"}


def _second_source_bars(source: str, symbol: str, start, end) -> pd.DataFrame | None:
    """The other vendor's recent bars for an independent check; None when it is unavailable.
    Called outside the breaker chain so a confirmation miss never disables the primary path."""
    try:
        bars = nasdaq.history(symbol, start, end, "stocks") if source == "nasdaq" else yahoo.history(symbol, start, end)[0]
        return prices.normalise(bars)
    except Exception:  # noqa: BLE001 - confirmation is additional evidence, not a prerequisite
        return None


def eligibility_check(observations: dict[str, tuple[float | None, float | None]], filters: dict) -> dict:
    """Does every observed (price, volume) pass the user's price/volume filter?

    Eligibility is only "confirmed" when all sources agree; any source below a
    threshold leaves it "unconfirmed" - the stock stays, but is not declared eligible."""
    if not (filters.get("min_price") or filters.get("min_volume")):
        return {"status": "not_applied", "observations": observations}
    verdicts = {name: ranking.passes(price, volume, filters) for name, (price, volume) in observations.items()
                if price is not None}
    return {"status": "confirmed" if verdicts and all(verdicts.values()) else "unconfirmed",
            "observations": observations, "failing": sorted(k for k, ok in verdicts.items() if not ok)}


def _benchmark_bars(index: str, proxy: str, start, end, *, days=5, is_intraday=False) -> tuple[pd.DataFrame, str, str | None, list[str]]:
    def validated(bars):
        frame = prices.normalise(bars)
        prices.validate_window(frame, symbol=index, days=days, trade_date=end,
                               is_intraday=is_intraday, outlier_move=1.0)
        return frame

    def from_yahoo():
        bars, _, _ = yahoo.history(index, start, end)
        return validated(bars), None

    def from_nasdaq_proxy():
        return validated(nasdaq.history(proxy, start, end, "etf")), proxy

    chain = first_success([("yahoo", from_yahoo), ("nasdaq", from_nasdaq_proxy)])
    frame, used_proxy = chain.value
    return frame, chain.source, used_proxy, chain.warnings


def _split_move(window: pd.DataFrame, split_checks: list, trade_date, screener_pct: float) -> dict | None:
    """The trade-date move once a split that day is adjusted, when the screener's figure is the unadjusted one."""
    split = next((c for c in split_checks if c.status == "adjusted" and c.date == str(trade_date)), None)
    if split is None or len(window) < 2:
        return None
    adjusted = float(window["close"].iloc[-1] / window["close"].iloc[-2] - 1) * 100
    unadjusted = ((1 + adjusted / 100) / split.ratio - 1) * 100
    if abs(unadjusted - screener_pct) >= abs(adjusted - screener_pct):
        return None                 # the screener already used the adjusted close
    return {"date": split.date, "ratio": split.ratio, "screener_pct": screener_pct, "adjusted_pct": round(adjusted, 4)}


def _successor(state: dict, symbol: str) -> dict | None:
    ranked = state.get("candidates") or []
    names = [c["symbol"] for c in ranked]
    return ranked[names.index(symbol) + 1] if symbol in names[:-1] else None


def _without_split_artifact(state: dict, symbol: str, successor: dict, split_move: dict, days: int, is_intraday: bool) -> dict:
    """State once a split-driven screener gain is ruled out: the next candidate takes its place."""
    selection = dict(state.get("selection") or {})
    selection["split_excluded"] = [*(selection.get("split_excluded") or []), {"symbol": symbol, **split_move}]
    update = {"candidates": [c for c in state.get("candidates") or [] if c["symbol"] != symbol], "selection": selection}
    if selection.get("primary_symbol") != symbol:
        return update
    selection["primary_symbol"] = successor["symbol"]
    old = state.get("ranking") or {}
    own = next((UNRESOLVED[f] for f in UNRESOLVED if f in (successor.get("flags") or [])), None)
    status = ranking.assess({"symbol": successor["symbol"], "pct_change": successor["pct_change"], "unresolved": own},
                            old.get("contenders") or [], is_intraday=is_intraday)
    contenders = {c["symbol"] for c in old.get("contenders") or []}
    status["ruled_out"] += [s for s in old.get("ruled_out") or [] if s not in contenders] + [symbol]
    flags = [f for f in successor.get("flags") or [] if f != "ranking_unverified"]
    flags += ["ranking_unverified"] if status["status"] == "uncertain" else []
    update.update(ranking=status, selected={**successor, "flags": flags, "days": days},
                  analysis={}, sentiment={}, datasets={}, charts=[])
    return update


@tool("get_price_history")
@agent_tool
def get_price_history(
    symbol: Annotated[str, Field(min_length=1, max_length=10, description="Ticker chosen from get_top_nasdaq_gainers candidates.")],
    days: Annotated[int | None, Field(ge=2, le=30, description="Trading days to analyse (default from config, normally 5).")] = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolResult:
    """Fetch, clean and validate daily prices for the chosen stock plus the S&P 500 and
    NASDAQ Composite benchmarks, and save them as datasets for execute_python.

    Call after choosing a candidate. Returns dataset ids (not the prices themselves), the
    data checks that ran (cross-source check of the day's move, splits/dividends, gaps) and
    any flags. If flags include data_conflict or possible_split_artifact, follow the advice.
    Prices before a split are restated when the bars still jump by the split factor; a stock
    whose gain was only the split is removed and the error names the candidate to analyse instead."""
    ctx = get_ctx()
    settings = ctx.settings
    state = state or {}
    symbol = symbol.strip().upper()
    if days is not None and days != settings.analysis.days:
        raise ToolError("window_scope_locked", f"the user requested {settings.analysis.days} trading days",
                        hint="Do not silently shorten or expand the requested window. Report insufficient history explicitly.")
    days = days or settings.analysis.days

    candidates = {c["symbol"]: c for c in state.get("candidates", [])}
    raw_top = state.get("raw_top") or {}
    if raw_top:
        candidates.setdefault(raw_top["symbol"], raw_top)
    if symbol not in candidates:
        raise ToolError(
            "unknown_symbol", f"{symbol} is not among the screener candidates.", retryable=False,
            hint=f"Choose one of {sorted(candidates)[:10]} or call get_top_nasdaq_gainers first.",
        )
    candidate = candidates[symbol]
    session = state.get("session") or market_session().as_dict()
    trade_date = pd.Timestamp(session["trade_date"]).date()
    start = sessions_back(trade_date, days + 12)
    checks = settings.data_checks

    # The stock and both benchmarks are independent requests: fetch them concurrently.
    benchmarks = settings.analysis.benchmarks
    with ThreadPoolExecutor(max_workers=1 + len(benchmarks)) as pool:
        stock_job = pool.submit(copy_context().run, _stock_bars, symbol, start, trade_date,
                                days=days, is_intraday=session["is_intraday"], outlier_move=checks.outlier_daily_move)
        bench_jobs = {index: pool.submit(copy_context().run, _benchmark_bars, index, proxy, start, trade_date,
                                        days=days, is_intraday=session["is_intraday"])
                      for index, proxy in benchmarks.items()}
        # Yahoo is tried first, so Nasdaq is the likely second vendor: fetch it alongside.
        nasdaq_job = pool.submit(copy_context().run, _second_source_bars, "nasdaq", symbol,
                                 sessions_back(trade_date, 3), trade_date)
    stock, actions, source, warnings = stock_job.result()
    flags = []
    if actions is None:
        flags.append("corporate_actions_unchecked")
        warnings.append("split and dividend data unavailable: no source returned corporate actions for this window")
        actions = []
    stock, split_checks = corporate_actions.adjust_for_splits(stock, actions)
    window, check = prices.validate_window(
        stock, symbol=symbol, days=days, trade_date=trade_date,
        is_intraday=session["is_intraday"], outlier_move=checks.outlier_daily_move,
    )
    check.processing += [f"bars before {c.date} restated for a {corporate_actions.split_name(c.ratio)} "
                         f"(prices / {c.ratio:g}, volume x {c.ratio:g})" for c in split_checks if c.status == "adjusted"]
    warnings += check.warnings
    flags += check.flags
    frames = {symbol: window}
    meta = {symbol: {"symbol": symbol, "source": source, "proxy_for": None}}

    for index in benchmarks:
        try:
            bench, bsource, used_proxy, bwarn = bench_jobs[index].result()
        except Exception as exc:
            warnings.append(f"{index} unavailable after validation and fallback: {exc}; omitted, stock analysis continues")
            continue
        bwindow, bcheck = prices.validate_window(
            bench, symbol=index, days=days, trade_date=trade_date,
            is_intraday=session["is_intraday"], outlier_move=1.0,
        )
        warnings += bwarn + bcheck.warnings
        if used_proxy:
            warnings.append(f"{index} unavailable; using {used_proxy} ETF as a proxy")
        if list(bwindow["date"]) != list(window["date"]):
            warnings.append(f"{index}: dates do not match the stock window; benchmark omitted")
            continue
        frames[index] = bwindow
        meta[index] = {"symbol": used_proxy or index, "source": bsource, "proxy_for": index if used_proxy else None}

    aligned, align_warnings = prices.align(frames)
    warnings += align_warnings
    for key, frame in aligned.items():
        if len(frame) != days + 1 or list(frame["date"]) != list(window["date"]):
            raise ToolError("missing_sessions", f"{key}: alignment changed the required window", retryable=True)

    screener_pct = candidate["pct_change"]
    split_move = _split_move(aligned[symbol], split_checks, trade_date, screener_pct)
    if split_move:
        warnings.append(f"the screener's {screener_pct:+.2f}% compares the close with the pre-split price; "
                        f"split-adjusted the day's move is {split_move['adjusted_pct']:+.2f}%")
        successor = _successor(state, symbol)
        if successor and split_move["adjusted_pct"] < successor["pct_change"]:
            raise ToolError(
                "split_artifact",
                f"{symbol}'s screener gain of {screener_pct:+.2f}% comes from a {corporate_actions.split_name(split_move['ratio'])}: "
                f"split-adjusted, the day's move was {split_move['adjusted_pct']:+.2f}%, below "
                f"{successor['symbol']}'s {successor['pct_change']:+.2f}%.",
                hint=f"{symbol} is not a top-gainer candidate and was removed. Call get_price_history for "
                     f"{successor['symbol']} (and get_news); it is now the top candidate, not a supplementary one.",
                update=_without_split_artifact(state, symbol, successor, split_move, days, session["is_intraday"]))
        # Still the leader after adjustment: compare like with like.
        screener_pct = ((1 + screener_pct / 100) * split_move["ratio"] - 1) * 100
    cross = compare_day_move(aligned[symbol], screener_pct, checks.cross_source_warn_pp, checks.cross_source_conflict_pp)
    flags += cross.flags
    if cross.status == "warn":
        warnings.append(cross.message)
    second = SECOND_SOURCE[source]
    second_bars = nasdaq_job.result() if second == "nasdaq" else _second_source_bars(second, symbol, sessions_back(trade_date, 3), trade_date)
    if second_bars is not None:
        second_bars = corporate_actions.adjust_for_splits(second_bars, actions)[0]   # each vendor is tested on its own bars
    confirmation = price_confirmation(stock, second_bars, trade_date, primary_source=source,
                                              secondary_source=second if second_bars is not None else None,
                                              conflict_pp=checks.cross_source_conflict_pp)
    flags += confirmation.flags
    if confirmation.status != "consistent":
        warnings.append(confirmation.message)
    observations = {"screener": (candidate.get("price"), candidate.get("volume")),
                    f"{source} daily": (float(window["close"].iloc[-1]), float(window["volume"].iloc[-1]))}
    if confirmation.volume:
        observations[f"{second} daily"] = (confirmation.close.get(second), confirmation.volume.get(second))
    eligible = eligibility_check(observations, (state.get("selection") or {}).get("filters") or {})
    if eligible["status"] == "unconfirmed":
        flags.append("eligibility_unconfirmed")
        seen = "; ".join(f"{name} price {price:,.2f}, volume {volume:,.0f}" for name, (price, volume) in observations.items()
                         if price is not None and volume is not None)
        warnings.append(f"filter eligibility unconfirmed: below the price/volume filter in {', '.join(eligible['failing'])} ({seen})")
    in_window = [a for a in actions if a.date >= str(window["date"].iloc[0])]
    if in_window:
        flags.append("corporate_action")
    advice = None
    # Only a split the bars could not test leaves the gain in doubt; a tested one was adjusted or did not apply.
    tested = {c.date for c in split_checks if c.status != "unknown"}
    if corporate_actions.possible_split_artifact([a for a in in_window if a.date not in tested], trade_date, candidate["pct_change"]):
        flags.append("possible_split_artifact")
        advice = "The gain may be a reverse-split artefact. Report the uncertainty; a different stock is supplementary, not the top gainer."
    elif "data_conflict" in flags:
        advice = ("The screener and price history disagree on the day's move. Call get_price_history once more "
                  "if it persists, report the discrepancy without claiming a verified ranking.")
    elif "source_conflict" in flags:
        advice = ("Two independent vendors disagree on the trade-date close. The report will state this; "
                  "do not describe the ranking as verified.")

    datasets = {}
    captured_at = datetime.now(timezone.utc).isoformat()
    for key, frame in aligned.items():
        ds = dataset_id(key)
        path = ctx.data_dir / f"{ds}.csv"
        frame.to_csv(path, index=False)
        datasets[ds] = {
            **meta[key], "dataset_id": ds, "path": str(path), "rows": len(frame),
            "first_date": str(frame["date"].iloc[0]), "last_date": str(frame["date"].iloc[-1]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trade_date": str(trade_date),
            "captured_at": captured_at,
        }

    analysis_key = hashlib.sha256(json.dumps({"symbol": symbol, "days": days, "trade_date": str(trade_date),
                                              "datasets": {k: v["sha256"] for k, v in datasets.items()}}, sort_keys=True).encode()).hexdigest()
    selected = {**candidate, "dataset_id": dataset_id(symbol), "days": days, "analysis_key": analysis_key,
                "trade_date": str(trade_date), "flags": sorted(set(candidate.get("flags", []) + flags))}
    data = {
        "selected": {"symbol": symbol, "pct_change": candidate["pct_change"], "trade_date": str(trade_date)},
        "datasets": [{k: v for k, v in d.items() if k != "path"} for d in datasets.values()],
        "columns": prices.COLUMNS,
        "checks": {
            "screener_vs_history": cross.message,
            "price_confirmation": confirmation.message,
            "filter_eligibility": eligible["status"],
            "corporate_actions": [a.describe() for a in in_window] or "none in window",
            "processing": check.processing,
        },
        "flags": sorted(set(flags)),
    }
    if advice:
        data["advice"] = advice
    return ToolResult(
        data=data,
        warnings=warnings,
        source=source,
        update={
            "selected": selected,
            "datasets": datasets,
            "checks": {
                "screener_vs_history": {**cross.__dict__, "same_vendor": source == candidate.get("source")},
                "price_confirmation": confirmation.__dict__,
                "filter_eligibility": eligible,
                "corporate_actions": [a.describe() for a in in_window],
                "dividends": [{"ex_date": a.date, "amount": a.value} for a in in_window if a.kind == "dividend"],
                "splits": [{"date": a.date, "ratio": a.value,
                            **next(({"status": c.status, "open": c.open, "previous_close": c.previous_close}
                                    for c in split_checks if c.date == a.date), {"status": "unknown"})}
                           for a in in_window if a.kind != "dividend"],
                "split_move": split_move,
                "processing": check.processing,
            },
            "sources": {"history": source, **{f"benchmark {k}": v["source"] for k, v in meta.items() if k != symbol}},
            "analysis": {},
            "sentiment": {},
            "charts": [],
        },
    )

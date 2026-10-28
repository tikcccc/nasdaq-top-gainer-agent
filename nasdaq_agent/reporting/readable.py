"""Reader-friendly report sentences, rendered deterministically from verified facts.

Wording is chosen by fixed thresholds on the verified numbers (for example "almost
all" only when the final session supplied at least 90% of the net move, "well ahead
of" only for a gap of ten percentage points or more), so it reads like an analyst
note while remaining a function of the data. No model text enters these sentences.
"""
from __future__ import annotations

import statistics
from datetime import date

from nasdaq_agent.reporting.claims import TOL, path_facts
from nasdaq_agent.reporting.placeholders import compact, pct_units

FLAT_BAND = 0.005          # |gap| within half a percentage point reads as "in line with"
WIDE_GAP = 0.10            # ten percentage points or more reads as "well ahead/behind"
SWINGS = ((0.05, "large"), (0.02, "moderate"), (0.0, "small"))   # daily volatility labels
HEAVY_VOLUME = 3.0         # at least three times the earlier average reads as "unusually heavy"
EXTREME_VOLUME = 10.0      # ten times or more reads as "an extreme spike"
NORMAL_VOLUME = (2 / 3, 1.5)   # within this band of the earlier average reads as "in line with"
OUTLIER_VOL = 0.5          # volatility without the final session under half the total is reported apart
SHOCK_MOVE = 0.20          # an earlier session moving 20% or more gets its own sentence
SMALL_INDEX_MOVE = 0.1     # both indexes moved under a tenth of the stock's move


def pct(value: float, signed: bool = False) -> str:
    """Use the card/subject formatter for prose too; magnitude unless signed."""
    shown = value if signed else abs(value)
    return pct_units(shown * 100, signed=signed)


def times(ratio: float) -> str:
    """Use the same precision in cards and prose: whole multiples above ten, tenths below."""
    if ratio < 10:
        return f"{ratio:.1f}"
    return f"{ratio:,.0f}"


def short_date(iso: str | None) -> str:
    if not iso:
        return "the trade date"
    d = date.fromisoformat(iso[:10])
    return f"{d.strftime('%b')} {d.day}"


def _moved(value: float, rose: str = "rose", fell: str = "fell") -> str:
    if pct(value) == "0.00%":       # "fell 0.00%" reads as a contradiction
        return "was flat"
    return f"{rose} {pct(value)}" if value > 0 else f"{fell} {pct(value)}"


def _and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _trough(returns: list[float]) -> tuple[float, int]:
    """Deepest close-to-peak drawdown on the indexed path, and the close it was reached at."""
    closes, peak, worst, at = [1.0], 1.0, 0.0, 0
    for r in returns:
        closes.append(closes[-1] * (1 + r))
        peak = max(peak, closes[-1])
        if closes[-1] / peak - 1 < worst:
            worst, at = closes[-1] / peak - 1, len(closes) - 1
    return worst, at


def _relation(gap: float) -> str:
    return ("in line with" if abs(gap) <= FLAT_BAND else
            ("well ahead of" if gap >= WIDE_GAP else "ahead of") if gap > 0 else
            ("well behind" if gap <= -WIDE_GAP else "behind"))


def stock_specific(analysis: dict) -> bool:
    """Both indexes moved under a tenth of a wide stock move, so one sentence covers both comparisons."""
    facts, bench = path_facts(analysis), analysis.get("benchmark") or {}
    if not facts or "GSPC" not in bench or "IXIC" not in bench:
        return False
    total = facts["window_return"]
    relations = {_relation(total - bench[key]) for key in ("GSPC", "IXIC")}
    return (len(relations) == 1 and relations.pop().startswith("well") and abs(total) >= WIDE_GAP
            and max(abs(bench["GSPC"]), abs(bench["IXIC"])) <= SMALL_INDEX_MOVE * abs(total))


def sentences(analysis: dict, *, symbol: str, trade_date: str | None, volume: dict | None = None,
              dividend: dict | None = None, shock: dict | None = None) -> dict[str, str]:
    """One sentence per finding kind that the verified data supports."""
    facts = path_facts(analysis)
    if not facts:
        return {}
    day = short_date(trade_date)
    n = len(facts["session_directions"])
    total, prior, final = facts["window_return"], facts["pre_final_return"], facts["final_session_return"]
    ups, downs = facts["up_sessions"], facts["down_sessions"]
    out: dict[str, str] = {}

    flats = facts["flat_sessions"]
    if ups == n or downs == n:
        days = f"{'rising' if ups == n else 'falling'} on all {n} days"
    else:
        days = "with " + _and([f"{count} {label} day{'s' if count != 1 else ''}"
                               for count, label in ((ups, "up"), (downs, "down"), (flats, "flat")) if count])
    out["window_return"] = f"Over the {n} sessions to {day}, {symbol} {_moved(total)}, {days}."

    move = "gain" if total > TOL else "loss"
    share = facts["final_share_of_net_change"]
    earlier = "the earlier session" if n == 2 else f"the {n - 1} earlier sessions together"
    if abs(total) <= TOL:
        out["final_session_share"] = "The window ended roughly flat, so no single session accounts for the net move."
    elif facts["final_accounts_for_all_net_change"]:
        out["final_session_share"] = (f"All of the net {move} came on {day}; the earlier sessions netted to roughly zero.")
    elif facts["final_accounts_for_majority_net_change"]:
        amount = "Almost all" if share >= 0.9 else "Most"
        out["final_session_share"] = (f"{amount} of the {n}-session {move} came on {day}, when the stock {_moved(final)}; "
                                      f"{earlier} {_moved(prior, 'added', 'lost')}.")
    elif facts["final_index_change"] * total < -TOL:
        out["final_session_share"] = (f"The {day} session ({pct(final, True)}) cut into the earlier "
                                      f"{pct(prior, True)} move.")
    elif abs(facts["final_index_change"]) <= TOL:
        out["final_session_share"] = f"The {day} session was flat; the earlier sessions account for the move."
    elif prior * facts["final_index_change"] < -TOL:
        out["final_session_share"] = (f"The {day} session ({pct(final, True)}) reversed the earlier "
                                      f"{n - 1} sessions' cumulative {pct(prior, True)} move.")
    else:
        out["final_session_share"] = (f"Both the earlier sessions ({pct(prior, True)}) and {day} "
                                      f"({pct(final, True)}) contributed to the {move}.")

    benches = {}
    for key, name in (("GSPC", "S&P 500"), ("IXIC", "NASDAQ Composite")):
        bench = (analysis.get("benchmark") or {}).get(key)
        if bench is None:
            continue
        benches[key] = (name, bench, _relation(total - bench))
        out[f"vs_{key}"] = (f"Its {n}-session return was {benches[key][2]} the {name}, "
                            f"which {_moved(bench)} over the same dates.")
    # Both comparisons in one sentence, for the body when neither is a key finding.
    if len(benches) == 2:
        (_, sp, sp_rel), (_, nq, nq_rel) = benches["GSPC"], benches["IXIC"]
        if sp_rel != nq_rel:
            out["vs_both"] = (f"Its {n}-session return was {sp_rel} the S&P 500, which {_moved(sp)}, "
                              f"and {nq_rel} the NASDAQ Composite, which {_moved(nq)}, over the same dates.")
        else:
            indexes = f"Over the same dates the S&P 500 {_moved(sp)} and the NASDAQ Composite {_moved(nq)}"
            out["vs_both"] = (f"{indexes}; {symbol}'s {pct(total)} {n}-session {move} was much larger than either index's move."
                              if stock_specific(analysis) else
                              f"{indexes}, so {symbol}'s {n}-session return was {sp_rel} both.")

    vol, drawdown = analysis.get("volatility_daily"), analysis.get("max_drawdown")
    returns = analysis["daily_returns"]
    if isinstance(vol, (int, float)):
        rest = statistics.stdev(returns[:-1]) if n > 2 else None
        if rest is not None and rest < statistics.stdev(returns) * OUTLIER_VOL:
            # One outsized session makes the window figure describe that day, not the typical one.
            out["volatility"] = (f"Daily sample volatility was {pct(rest)} over the {n - 1} earlier returns "
                                 f"and {pct(vol)} over all {n}; neither figure is annualized.")
        else:
            label = next(word for floor, word in SWINGS if vol >= floor)
            out["volatility"] = f"Day-to-day swings were {label}: {pct(vol)} daily sample volatility across {n} returns, not annualized."
    if isinstance(drawdown, (int, float)):
        if drawdown > -TOL:
            out["drawdown"] = "The stock never closed below an earlier high in the window."
        else:
            worst, at = _trough(returns)
            when = ("" if abs(worst - drawdown) > 1e-4 else
                    f", reached on {day}" if at == n else f", reached before {day}")
            out["drawdown"] = f"The deepest close-to-close pullback from a closing high in the window was {pct(drawdown)}{when}."
    if volume:
        ratio = volume["ratio"]
        average = f"the previous {volume['sessions']}-session average of {compact(volume['shares'] / ratio)} shares a day"
        tail = (" — an extreme spike in trading" if ratio >= EXTREME_VOLUME else
                " — unusually heavy trading" if ratio >= HEAVY_VOLUME else "")
        out["volume_spike"] = (f"{day} volume was in line with {average}." if NORMAL_VOLUME[0] <= ratio <= NORMAL_VOLUME[1] else
                               f"{day} volume was {times(ratio)} times {average}{tail}.")
    if shock:
        when = short_date(shock["date"])
        move = f"{'rose' if shock['return'] > 0 else 'fell'} {pct(abs(shock['return']))} on {when}"
        coverage = "a fetched headline is dated that day, but " if shock.get("ref") else ""
        out["prior_shock"] = f"{symbol} {move}; {coverage}the cause has not been verified for this report."
    if dividend and dividend.get("ref"):
        # Headline-sourced: the amount is the source's claim, so the sentence attributes it and cites the headline.
        out["dividend"] = (f"The {dividend['publisher']} headline [{dividend['ref']}] states a "
                           f"${dividend['amount']:,.2f} per-share dividend, equal to "
                           f"{pct(dividend['amount'] / dividend['prior_close'])} of {symbol}'s "
                           f"{short_date(dividend['prior_date'])} close of ${dividend['prior_close']:,.2f}, "
                           f"the last close before it was published; the share price typically drops by about "
                           f"the dividend on the ex-dividend date.")
    elif dividend:
        ex = short_date(dividend["ex_date"])
        line = f"A ${dividend['amount']:,.2f} per-share dividend went ex on {ex}"
        if dividend.get("prior_close"):
            line += (f", equal to {pct(dividend['amount'] / dividend['prior_close'])} of the "
                     f"{short_date(dividend['prior_date'])} close of ${dividend['prior_close']:,.2f}")
        if dividend.get("in_returns"):
            # Closes are split-adjusted only, so the payout coming off the price is inside that day's return.
            line += f"; these closes are not dividend-adjusted, so the {ex} return includes the payout coming off the price"
        out["dividend"] = line + "."
    return out


TOPICS = {"window_return": "path", "final_session_share": "path", "vs_sp500": "benchmark",
          "vs_nasdaq": "benchmark", "volume_spike": "volume", "dividend": "dividend"}


def headline(kinds: list[str], analysis: dict, *, symbol: str, trade_date: str | None, session_move: float | None,
             volume: dict | None = None, dividend: dict | None = None, index_day: dict | None = None) -> str:
    """One-line summary: the day's move plus the most important fact Key finding 1 does not state.

    The fact comes from the agent's later findings in order, then from the price path,
    a dividend, heavy volume or the benchmarks, skipping finding 1's own topic.
    """
    facts = path_facts(analysis)
    day = short_date(trade_date)
    n = len(facts.get("session_directions") or [])
    total = facts.get("window_return", 0.0)
    today = f"{symbol} {_moved(session_move, 'jumped' if (session_move or 0) >= 0.2 else 'rose', 'dropped' if (session_move or 0) <= -0.2 else 'fell')} on {day}" \
        if session_move is not None else f"{symbol} traded on {day}"
    if kinds and kinds[0] == "data_limitation":
        return f"{today}, but the data limitations below apply before relying on this result."
    if not facts:
        return f"{today}."
    if abs(total) <= TOL and (session_move is None or abs(session_move) <= TOL):
        return f"{symbol} was flat on {day} and over the last {n} sessions."
    move = "gain" if total > 0 else "loss"
    bench = analysis.get("benchmark") or {}

    def clause(kind: str) -> str | None:
        if TOPICS.get(kind) == "path":
            # A day move next to a near-equal window move reads as two competing numbers; say how they relate.
            if facts["final_accounts_for_majority_net_change"] and session_move is not None:
                amount = "almost all" if facts["final_share_of_net_change"] >= 0.9 else "most"
                return f" — {amount} of its {pct(total)} {move} over the last {n} sessions"
            if facts["final_accounts_for_all_net_change"]:
                return f"; all of its {n}-session {move} came that day"
            return f" and {_moved(total)} over the last {n} sessions"
        if TOPICS.get(kind) == "benchmark":
            # Same span on both sides: the indexes' moves on the trade date beside the stock's day move.
            named = [(name, (index_day or {}).get(key)) for key, name in (("GSPC", "the S&P 500"), ("IXIC", "the NASDAQ Composite"))]
            named = [(name, move) for name, move in named if move is not None]
            if named and session_move is not None:
                return ", while " + _and([f"{name} {_moved(move)}" for name, move in named]) + " that day"
            keys = ("GSPC", "IXIC") if stock_specific(analysis) else ("IXIC",) if kind == "vs_nasdaq" else ("GSPC",)
            names = {"GSPC": "the S&P 500", "IXIC": "the NASDAQ Composite"}
            if not all(k in bench for k in keys):
                return None
            return (f"; over the last {n} sessions it {_moved(total)}, while "
                    + _and([f"{names[k]} {_moved(bench[k])}" for k in keys]))
        if kind == "volume_spike" and volume:
            return f", with volume {times(volume['ratio'])} times its recent average"
        if kind == "dividend" and dividend and dividend.get("ref"):
            # No citation in the brief: the body's dividend sentence cites the headline.
            return (f"; a ${dividend['amount']:,.2f} per-share dividend reported in a headline equals "
                    f"{pct(dividend['amount'] / dividend['prior_close'])} of the prior close")
        if kind == "dividend" and dividend:
            share = f", {pct(dividend['amount'] / dividend['prior_close'])} of the prior close," if dividend.get("prior_close") else ""
            return f"; a ${dividend['amount']:,.2f} dividend{share} went ex on {short_date(dividend['ex_date'])}"
        return None

    heavy = ["volume_spike"] if volume and volume["ratio"] >= HEAVY_VOLUME else []
    lead = TOPICS.get(kinds[0]) if kinds else None
    for kind in [*kinds[1:], "final_session_share", "dividend", *heavy, "vs_sp500"]:
        if TOPICS.get(kind) != lead and (text := clause(kind)):
            return f"{today}{text}."
    return f"{today}."

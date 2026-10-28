"""Numbers in the email come from code, never from the LLM.

The agent writes its narrative with ``{{placeholders}}``; this module builds the
values from validated state and rejects narratives that contain literal figures.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from nasdaq_agent.data.corporate_actions import split_name
from nasdaq_agent.infra.schemas import UNVERIFIED_FLAGS
from nasdaq_agent.reporting import claims

PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z_][\w.]*)\s*}}")
CITATION = re.compile(r"\[([nd]\d+)\]")  # [n1] headline, [d2] data source
# Literal figures the LLM must not write: percentages, dollar amounts, decimals.
RAW_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s?%|\$\s?\d[\d,]*(?:\.\d+)?|(?<![\w.])\d+\.\d+")


FLAG_LABELS = {
    "derivative": "warrant/right/unit",
    "penny": "sub-$1",
    "low_volume": "thinly traded",
    "micro_cap": "micro-cap",
    "inconsistent": "inconsistent quote",
    "outlier": "outlier move",
    "corporate_action": "corporate action in window",
    "corporate_actions_unchecked": "split/dividend data unavailable",
    "data_conflict": "sources disagree",
    "possible_split_artifact": "possible split artefact",
    "source_conflict": "vendors disagree on the close",
    "eligibility_unconfirmed": "filter eligibility unconfirmed",
    "unverified_quote": "quote not verified",
    "ranking_unverified": "ranking not confirmed",
    "near_threshold": "near a filter threshold",
}


CATALYST_LABELS = {
    "earnings": "earnings / guidance",
    "regulatory_clinical": "regulatory or clinical news",
    "deal": "deal or partnership",
    "financing": "financing / dilution",
    "corporate_action": "corporate action",
    "sector_macro": "sector or macro move",
    "speculation": "no clear news (momentum)",
}


def pct_units(percent: float | None, signed: bool = True) -> str:
    """Percent value -> text. Big moves get one decimal and thousands separators (+2,800.0%)."""
    if percent is None:
        return "n/a"
    digits = 1 if abs(percent) >= 100 else 2
    return f"{percent:{'+' if signed else ''},.{digits}f}%"


def pct(fraction: float | None, digits: int | None = None, signed: bool = True) -> str:
    if fraction is None:
        return "n/a"
    if digits is not None:
        return f"{fraction * 100:{'+' if signed else ''},.{digits}f}%"
    return pct_units(fraction * 100, signed)


def points(fraction: float | None) -> str:
    return f"{fraction * 100:+,.2f} pp" if fraction is not None else "n/a"


def money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def compact(value: float | None) -> str:
    """41_847_594 -> '41.8M'."""
    if not value:
        return "n/a"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= size:
            return f"{value / size:.1f}{unit}"
    return f"{value:,.0f}"


def human_date(iso: str | None) -> str:
    """'2026-10-23' -> 'Fri, Oct 23, 2026'."""
    try:
        return date.fromisoformat(str(iso)).strftime("%a, %b %d, %Y").replace(" 0", " ")
    except ValueError:
        return str(iso or "n/a")


def flag_text(flags: list[str] | None) -> str:
    return ", ".join(FLAG_LABELS.get(f, f) for f in flags or []) or "none"


def build_context(state: dict, *, report_facts=None) -> dict[str, Any]:
    """Formatted values available to the narrative template (nested for Jinja)."""
    selected = state.get("selected") or {}
    session = state.get("session") or {}
    analysis = state.get("analysis") or {}
    if report_facts is None and selected.get("analysis_key"):
        from nasdaq_agent.reporting.facts import build_report_facts
        report_facts = build_report_facts(state)
    if report_facts is not None:
        analysis = report_facts.analysis
    sentiment = state.get("sentiment") or {}
    raw_top = state.get("raw_top") or {}
    selection = state.get("selection") or {}
    filters = selection.get("filters") or {}
    criteria = ["common stocks" if filters.get("exclude_derivatives", True) else "all listed instruments"]
    criteria += ([f"price >= ${filters['min_price']:g}"] if filters.get("min_price") else [])
    criteria += ([f"volume >= {filters['min_volume']:,} shares"] if filters.get("min_volume") else [])
    criteria += ([f"market cap >= ${filters['min_market_cap']:g}"] if filters.get("min_market_cap") else [])
    label = selection.get("label", "Top NASDAQ common-stock gainer")
    if selection.get("primary_symbol") and selected.get("symbol") != selection["primary_symbol"]:
        label = "Supplementary NASDAQ stock analysis"
    if UNVERIFIED_FLAGS & set(selected.get("flags") or []):
        label = "Unverified NASDAQ gainer candidate"

    ctx: dict[str, Any] = {
        "symbol": selected.get("symbol", "n/a"),
        "company": selected.get("name", ""),
        "trade_date": human_date(session.get("trade_date")),
        "pct_change": pct_units(selected.get("pct_change")),
        "price": money(report_facts.price if report_facts is not None else selected.get("price")),
        "volume": compact(report_facts.shares if report_facts is not None else selected.get("volume")),
        # The screener supplies no dated, split-reconciled share count. Its cap is
        # unverified regardless of turnover; retain the raw value in the run record.
        "market_cap": "n/a",
        "days": str(selected.get("days") or analysis.get("days") or "n/a"),
        "selection_label": label,
        "selection_criteria": "; ".join(criteria),
    }
    if report_facts is not None and report_facts.day_return is not None:
        ctx["pct_change"] = pct(report_facts.day_return)
    if analysis:
        returns = analysis.get("daily_returns") or []
        bench = analysis.get("benchmark") or {}
        ctx["metrics"] = {
            "avg_daily_return": pct(analysis.get("avg_daily_return")),
            "cumulative_return": pct(analysis.get("cumulative_return")),
            "trend": analysis.get("trend", "n/a"),
            "volatility_daily": pct(analysis.get("volatility_daily"), signed=False),
            "volatility_annualized": pct(analysis.get("volatility_annualized"), 0, signed=False),
            "max_drawdown": pct(analysis.get("max_drawdown")),
            "excess_return": pct(analysis.get("excess_return")),
            "excess_return_pp": points(analysis.get("excess_return")),
            "gspc_return": pct(bench.get("GSPC")),
            "ixic_return": pct(bench.get("IXIC")),
            "best_day": pct(max(returns)) if returns else "n/a",
            "worst_day": pct(min(returns)) if returns else "n/a",
        }
        ctx["path"] = claims.text(analysis)
        count = len([h for h in (state.get("headlines") or {}).values()
                     if (not h.get("symbol") or h.get("symbol") == selected.get("symbol"))
                     and (not h.get("trade_date") or h.get("trade_date") == session.get("trade_date"))])
        ctx["news"] = {"coverage": (
            f"The configured sources returned {count} headline{'s' if count != 1 else ''} for this stock in the requested lookback. "
            "This is a fetched subset, not an exhaustive record of public news."
            if count else "The configured sources returned no headlines for this stock in the requested lookback. "
            "This does not establish that no public news or catalyst exists.")}
    if sentiment and sentiment.get("method") != "none":
        ctx["sentiment"] = {
            "label": sentiment.get("label", "n/a").lower(),
            "score": f"{sentiment.get('score', 0):.1f}/4",
            "confidence": f"{(sentiment.get('confidence') or 0) * 100:.0f}%",
            "catalyst": CATALYST_LABELS.get(sentiment.get("catalyst") or "", sentiment.get("catalyst") or "n/a"),
            "dilution_risk": f"{(sentiment.get('dilution_risk') or 0) * 100:.0f}%",
        }
    if raw_top and raw_top.get("symbol") != selected.get("symbol"):
        ctx["raw_top"] = {
            "symbol": raw_top["symbol"],
            "pct_change": pct_units(raw_top.get("pct_change")),
            "flags": flag_text(raw_top.get("flags")),
        }
    ctx["split_excluded"] = [{"symbol": e["symbol"], "screener_pct": pct_units(e["screener_pct"]),
                              "adjusted_pct": pct_units(e["adjusted_pct"]), "split": split_name(e["ratio"])}
                             for e in selection.get("split_excluded") or []]
    return ctx


# Context only the email template may use; the agent's narrative must not (the method note covers it).
TEMPLATE_ONLY = ("raw_top", "split_excluded")


def narrative_context(ctx: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in ctx.items() if k not in TEMPLATE_ONLY}


def flatten(ctx: dict[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in ctx.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = str(value)
    return out


def raw_numbers(text: str) -> list[str]:
    stripped = CITATION.sub("", PLACEHOLDER.sub("", text))
    return [m.group(0) for m in RAW_NUMBER.finditer(stripped)]


def problems(text: str, available: dict[str, str]) -> list[str]:
    """Static checks on the narrative: unknown placeholders, literal figures, template logic."""
    issues = []
    if "{%" in text or "{#" in text:
        issues.append("template statements ({% %}) are not allowed; use only {{placeholders}}")
    unknown = sorted({p for p in PLACEHOLDER.findall(text) if p not in available})
    if unknown:
        issues.append(f"unknown placeholders {unknown}")
    literals = raw_numbers(text)
    if literals:
        issues.append(f"literal figures {literals[:6]} found; replace each with a placeholder")
    return issues

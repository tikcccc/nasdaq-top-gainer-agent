"""A closed report contract: models select evidence, code writes the report.

No input field accepts prose. The agent chooses up to three verified findings
and which headlines to quote; code writes every sentence
from verified data, so emphasis is the model's judgement and wording is not.
Headlines are attributed source claims, never proof of price causation. This
boundary does not certify upstream source truth.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from nasdaq_agent.agent.evidence import analysis_problems
from nasdaq_agent.data import corporate_actions
from nasdaq_agent.infra.schemas import UNVERIFIED_FLAGS
from nasdaq_agent.reporting import announcements, claims, placeholders, readable
from nasdaq_agent.reporting.facts import build_report_facts
from nasdaq_agent.reporting.dates import news_date as _news_date
from nasdaq_agent.reporting.version import REPORT_VERSION

NewsId = Annotated[str, Field(pattern=r"^n[1-9]\d*$", max_length=20)]

FINDING_KINDS = {
    "window_return": "session directions and cumulative return over the window",
    "final_session_share": "how much of the window move came from the final session",
    "vs_sp500": "window return against the S&P 500 over the same dates",
    "vs_nasdaq": "window return against the NASDAQ Composite over the same dates",
    "volatility": "daily sample volatility",
    "drawdown": "maximum drawdown in the window",
    "volume_spike": "trade-date volume against the average of the earlier sessions",
    "prior_shock": "an earlier single-session move of at least 20%; automatically retained, with news dates matched in New York time",
    "dividend": ("a dividend against the prior close: one that went ex in the window, else the per-share "
                 "amount a selected headline states (attributed to it)"),
    "news": "a quoted, attributed headline (ref = its news ID)",
    "data_limitation": "the most material data limitation of this report",
}


class KeyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal[tuple(FINDING_KINDS)]  # type: ignore[valid-type]
    ref: NewsId | None = Field(None, description="News ID, only for kind 'news'.")


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    selected_news_ids: list[NewsId] = Field(default_factory=list, max_length=3,
        description="Up to three current-stock news IDs, in priority order; start with one. No prose.")
    selected_risk_evidence_ids: list[NewsId] = Field(default_factory=list, max_length=2,
        description="Optional source disclosures to quote as limitations; IDs only, no inferred risk or motive.")
    include_data_limitations: StrictBool = Field(default=True,
        description="Retained for compatibility. Material data limitations are always shown; no generic scope note is added.")
    key_findings: list[KeyFinding] | None = Field(None, min_length=1, max_length=3,
        description="1-3 findings, most important first. Reserve space for material data limitations. Kinds only.")


class ReportInputError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceReport:
    narrative: str
    citations: list[str]
    selection: dict
    quoted_ids: list[str]
    omitted_ids: list[str]
    word_count: int
    key_findings: list[dict]
    findings_source: str                # "agent" or "default"
    version: str = REPORT_VERSION


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", re.sub(r"\*\*|\[[nd]\d+\]", "", text)))


def current_headlines(state: dict) -> dict:
    selected, session = state.get("selected") or {}, state.get("session") or {}
    return {key: item for key, item in (state.get("headlines") or {}).items()
            if item.get("symbol") == selected.get("symbol")
            and item.get("trade_date") == session.get("trade_date")}


def _prior_shock(report_facts, available: dict) -> dict | None:
    """The largest earlier-session move when it is outsized, with any fetched headline dated that day."""
    returns = report_facts.analysis.get("daily_returns") or []
    stock = report_facts.stock
    if len(returns) < 2 or stock is None or len(stock) < len(returns) + 1:
        return None
    dates = stock["date"].astype(str).str[:10].tolist()[-len(returns):]
    i = max(range(len(returns) - 1), key=lambda k: abs(returns[k]))
    if abs(returns[i]) + 1e-12 < readable.SHOCK_MOVE:
        return None
    ref = next((hid for hid, h in available.items() if _news_date(h.get('published')) == dates[i]), None)
    return {"date": dates[i], "return": returns[i], "ref": ref}


def _identity(value) -> str:
    # External names must not insert report formatting/citations.
    return " ".join(str(value).split()).translate(str.maketrans({"[": "［", "]": "］", "*": "＊"}))


def _quote(hid: str, item: dict) -> str | None:
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ReportInputError(f"{hid}: source title is empty")
    # Unsafe markup is still visible, HTML-escaped, in the source list. Never
    # reinterpret it as report structure, templating or a second citation.
    if re.search(r"\{[{%#]|\*\*|\[[nd]\d+\]|[\r\n]", title):
        return None
    publisher = _identity(item.get("publisher") or "The fetched source")
    published = _news_date(item.get('published'))
    when = f" ({published})" if published else ""
    return f'{publisher}{when} published the headline: “{title}” [{hid}].'


def _limits(state: dict, analysis: dict, *, compact: bool = False, report_facts=None) -> list[str]:
    flags = set((state.get("selected") or {}).get("flags") or [])
    out = []
    if (state.get("session") or {}).get("is_intraday"):
        out.append("The market session was still open; the quote is intraday and not final [d1].")
    if report_facts is not None and report_facts.as_of_date != report_facts.trade_date:
        out.append(f"The latest price bar is dated {report_facts.as_of_date}; no price bar for "
                   f"the requested {report_facts.trade_date} session is available [d2].")
    # Every reason the ranking is unconfirmed shares one sentence, so several problems
    # at once cannot push the mandatory limits past the word budget.
    unresolved = ", ".join(_identity(u["symbol"]) for u in ((state.get("ranking") or {}).get("unresolved") or [])[:3])
    higher = "rows that could rank higher" + (f" ({unresolved})" if unresolved else "") + " could not be verified"
    reasons = [short if compact else text for flag, text, short in (
        ("data_conflict", "the quote and price history disagree", "quote vs history conflict"),
        ("possible_split_artifact", "a possible split artefact is unresolved", "possible split artefact"),
        ("stale", "the selected quote is flagged as stale", "stale quote"),
        ("source_conflict", "two price vendors disagree on the trade-date close", "vendor price conflict"),
        ("unverified_quote", "the screener quote could not be matched to this session's price history", "unmatched quote"),
        ("eligibility_unconfirmed", "sources disagree on whether the stock meets the price/volume filter",
         "filter eligibility unconfirmed"),
        ("ranking_unverified", higher, "unverified higher rows" + (f" ({unresolved})" if unresolved else "")),
    ) if flag in flags]
    if reasons:
        out.append(f"The ranking remains unverified and this is not a confirmed #1: {'; '.join(reasons)} [d1][d2].")
    if any(k not in (analysis.get("benchmark") or {}) for k in ("GSPC", "IXIC")):
        out.append("At least one matching benchmark window is unavailable; comparisons cover available data only [d2].")
    for index, name, ref in (("^GSPC", "S&P 500", "d3"), ("^IXIC", "NASDAQ Composite", "d4")):
        proxy = next((m for m in (state.get("datasets") or {}).values() if m.get("proxy_for") == index), None)
        if proxy:
            out.append(f"The {name} comparison uses {_identity(proxy['symbol'])} ETF prices as a proxy [{ref}].")
    checks = state.get("checks") or {}
    splits = checks.get("splits") or []
    for split in splits:
        out.append(_split_sentence(split, announcement=announcements.for_split(state, split)))
    # Dividends have their own sentence; structured splits are described above.
    if not splits and any("dividend" not in action for action in checks.get("corporate_actions") or []):
        out.append("Corporate-action data records an event within the price window [d2].")
    unchecked = "corporate_actions_unchecked" in flags
    # A headline about a split is a prompt to check the split data, not evidence of the split itself.
    split_news = next((hid for hid, item in current_headlines(state).items()
                       if isinstance(item.get("title"), str) and _SPLIT_NEWS.search(item["title"])), None)
    if split_news and not splits:
        out.append(f"A fetched headline reports a stock split [{split_news}]" +
                   (", and split data could not be checked; returns around its effective date may be distorted [d2]."
                    if unchecked else ", but the split data for this price window shows none; it may not have taken effect yet [d2]."))
    elif unchecked:
        out.append("Split and dividend data could not be checked for this price window [d2].")
    return out


def _split_sentence(split: dict, *, announcement: tuple[str, dict] | None = None) -> str:
    """A short event statement plus the material price-basis limitation."""
    what = corporate_actions.split_name(split["ratio"])
    day = readable.short_date(split["date"])
    event = (f"The company announced a {what} effective {day} [{announcement[0]}]"
             if announcement else f"Price data records a {what} on {day} [d2]")
    if split.get("status") == "adjusted":
        return f"{event}; earlier prices and volumes were restated [d2]."
    return f"{event}; price adjustment remains unconfirmed [d2]."


_SPLIT_NEWS = re.compile(r"\b(reverse\s+(?:stock\s+)?split|stock\s+split|share\s+consolidation)\b", re.I)


_PER_SHARE = re.compile(r"\$\s?(\d{1,4}(?:\.\d{1,4})?)\s*(?:per|a|/)\s*(?:common\s+|ordinary\s+)?share\b", re.I)
_DIVIDEND = re.compile(r"\bdividends?\b", re.I)
# A headline that withdraws, cuts or denies a dividend does not state one being paid.
_NOT_PAID = re.compile(r"\b(no|not|won't|will not|suspend\w*|cancel\w*|omit\w*|eliminat\w*|cut\w*|reduc\w*|halt\w*)\b", re.I)


def headline_dividend(title: str) -> float | None:
    """The one per-share amount a dividend headline states, or None when absent or ambiguous."""
    if not _DIVIDEND.search(title) or _NOT_PAID.search(title):
        return None
    amounts = {float(m) for m in _PER_SHARE.findall(title)}
    return amounts.pop() if len(amounts) == 1 and next(iter(amounts)) > 0 else None


def _close_before(frame: pd.DataFrame, published, trade_iso: str | None) -> tuple[str, float] | None:
    """The last close before a headline was published: a close at 16:00 ET counts once it has passed."""
    dates = frame["date"].astype(str).str[:10].tolist()
    cutoff, same_day = str(trade_iso)[:10], False
    if isinstance(published, str) and published:
        try:
            stamp = pd.Timestamp(published)
            stamp = stamp.tz_convert("America/New_York") if stamp.tzinfo else stamp
            cutoff, same_day = stamp.date().isoformat(), stamp.hour >= 16
        except ValueError:
            pass
    prior = max((i for i, d in enumerate(dates) if d < cutoff or (same_day and d == cutoff)), default=None)
    return (dates[prior], float(frame["close"].iloc[prior])) if prior is not None else None


def _dividend_facts(state: dict, cited_ids: list[str], available: dict) -> dict | None:
    """Corporate-action data first; otherwise the first cited headline that states a per-share dividend."""
    selected = state.get("selected") or {}
    meta = (state.get("datasets") or {}).get(selected.get("dataset_id", ""))
    if not meta:
        return None
    paid = (state.get("checks") or {}).get("dividends") or []
    frame = pd.read_csv(meta["path"])
    if paid:
        return _paid_dividend(paid, frame)
    for hid in cited_ids:
        item = available[hid]
        title = item.get("title")
        amount = headline_dividend(title) if isinstance(title, str) else None
        prior = _close_before(frame, item.get("published"), (state.get("session") or {}).get("trade_date")) if amount else None
        if prior:
            return {"amount": amount, "prior_date": prior[0], "prior_close": prior[1], "ref": hid,
                    "publisher": _identity(item.get("publisher") or "a fetched source")}
    return None


def _paid_dividend(paid: list[dict], frame: pd.DataFrame) -> dict:
    """The latest in-window dividend from corporate-action data, against the close before its ex-date."""
    latest = max(paid, key=lambda d: d["ex_date"])
    dates = frame["date"].astype(str).str[:10].tolist()
    prior = max((i for i, d in enumerate(dates) if d < latest["ex_date"]), default=None)
    return {"amount": float(latest["amount"]), "ex_date": latest["ex_date"],
            "prior_close": float(frame["close"].iloc[prior]) if prior is not None else None,
            "prior_date": dates[prior] if prior is not None else None,
            "in_returns": dates[0] < latest["ex_date"] <= dates[-1]}


def _material(state: dict) -> bool:
    flags = set((state.get("selected") or {}).get("flags") or [])
    return bool(flags & UNVERIFIED_FLAGS) or bool((state.get("session") or {}).get("is_intraday"))


FINDINGS_FORMAT = 'key_findings must be a list such as [{"kind": "vs_sp500"}, {"kind": "news", "ref": "n1"}]'


def normalize_findings(value):
    """Accept the shapes models actually send - a JSON string, or "kind" / "news:n1" items -
    and turn them into the strict structure. Validation itself is unchanged."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, list):
        return value
    out = []
    for item in value:
        if hasattr(item, "model_dump"):
            item = item.model_dump(exclude_none=True)
        if isinstance(item, str):
            kind, _, ref = item.strip().partition(":")
            item = {"kind": kind.strip(), **({"ref": ref.strip()} if ref.strip() else {})}
        out.append(item)
    return out


def build(state: dict, args: dict, settings) -> EvidenceReport:
    if args.get("key_findings") is not None:
        args = {**args, "key_findings": normalize_findings(args["key_findings"])}
    try:
        chosen = EvidenceSelection.model_validate(args)
    except ValidationError as exc:
        # Do not echo model-supplied prose or untrusted input into another prompt.
        fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
        raise ReportInputError("structured_report_required: only evidence-selection fields are accepted; invalid fields: "
                               + ", ".join(fields) + ("; " + FINDINGS_FORMAT if any(f.startswith("key_findings") for f in fields) else "")) from exc
    if not state.get("analysis"):
        raise ReportInputError("no verified analysis; execute_python must succeed before sending")
    issues = analysis_problems(state)
    if issues:
        raise ReportInputError("stale_analysis: " + "; ".join(issues))
    selected = state["selected"]
    if selected.get("trade_date") != (state.get("session") or {}).get("trade_date"):
        raise ReportInputError("selected stock belongs to another trade date")
    if len(set(chosen.selected_news_ids)) != len(chosen.selected_news_ids) or len(set(chosen.selected_risk_evidence_ids)) != len(chosen.selected_risk_evidence_ids):
        raise ReportInputError("duplicate evidence IDs; select each source once per list")
    available = current_headlines(state)
    ids = list(dict.fromkeys(chosen.selected_news_ids + chosen.selected_risk_evidence_ids))
    if any(hid not in available for hid in ids):
        raise ReportInputError("news evidence must belong to the current stock and trade date; fetch news again or select current IDs")
    sentiment = state.get("sentiment") or {}
    if sentiment and sentiment.get("method") != "none":
        if sentiment.get("symbol") != selected["symbol"] or sentiment.get("trade_date") != selected["trade_date"] or any(hid not in available for hid in sentiment.get("headline_ids") or []):
            raise ReportInputError("sentiment belongs to another stock/date or stale headlines; analyze_sentiment again")

    report_facts = build_report_facts(state)
    analysis = report_facts.analysis
    path = claims.text(analysis)
    if not path:
        raise ReportInputError("cannot calculate the verified price path")
    ctx = placeholders.build_context(state, report_facts=report_facts)
    symbol = _identity(ctx["symbol"])
    trade_iso = report_facts.as_of_date
    source = "agent" if chosen.key_findings else "default"
    findings = [(f.kind, f.ref) for f in chosen.key_findings] if chosen.key_findings else \
        [("final_session_share", None)]
    requested_findings = list(findings)
    if len(set(findings)) != len(findings):
        raise ReportInputError("duplicate key findings; select each finding once")
    for kind, ref in findings:
        if (kind == "news") != (ref is not None):
            raise ReportInputError("key findings: 'ref' is required for kind 'news' and not allowed for other kinds")
        if kind == "news" and ref not in available:
            raise ReportInputError(f"key finding news ref {ref} is not a current-stock, current-date headline")
    featured = {kind for kind, _ in findings}
    if chosen.key_findings and _material(state) and "data_limitation" not in featured:
        raise ReportInputError("material data limitations exist (unverified ranking/eligibility/price or intraday data); "
                               "include a 'data_limitation' key finding")
    # Featured news is quoted in the findings, so it does not need to be repeated in the disclosures.
    ids = list(dict.fromkeys([ref for kind, ref in findings if kind == "news"] + ids))
    volume, dividend = report_facts.volume, _dividend_facts(state, ids, available)
    shock = _prior_shock(report_facts, available)
    # News enters a finding only when there is space. Remaining titles stay in Sources.
    if not chosen.key_findings:
        if dividend:
            findings.append(("dividend", None))
        news = next((hid for hid in ids if _quote(hid, available[hid]) is not None
                     and word_count(_quote(hid, available[hid])) <= 65), None)
        if news:
            findings.append(("news", news))
        if shock:
            findings.append(("prior_shock", None))
        if volume and volume['ratio'] >= readable.HEAVY_VOLUME:
            findings.append(("volume_spike", None))
        findings = findings[:3]
    mandatory = _limits(state, analysis, report_facts=report_facts)
    # Reserve a shock slot even when the model supplied all three findings.
    # A material limitation and the shock cannot displace each other.
    if shock and 'prior_shock' not in {kind for kind, _ in findings}:
        if len(findings) == 3:
            drop = next(i for i in range(len(findings) - 1, -1, -1) if findings[i][0] != 'data_limitation')
            findings.pop(drop)
        findings.append(('prior_shock', None))
    if mandatory and "data_limitation" not in {kind for kind, _ in findings}:
        if len(findings) == 3:
            drop = next(i for i in range(len(findings) - 1, -1, -1) if findings[i][0] != 'prior_shock')
            findings.pop(drop)
        findings.append(("data_limitation", None))
    plain = readable.sentences(analysis, symbol=symbol, trade_date=trade_iso, volume=volume, dividend=dividend,
                                shock=shock)

    def cited(sentence: str, refs: str) -> str:
        return sentence and sentence.rstrip(".") + f" {refs}."

    # A stock-specific move needs no index-by-index comparison: one sentence covers both.
    combined = cited(plain["vs_both"], "[d2][d3][d4]") if readable.stock_specific(analysis) and "vs_both" in plain else ""

    def featured_sentence(kind: str, ref: str | None, limits: list[str]) -> str:
        sentences = {"window_return": cited(plain.get("window_return", ""), "[d2]"),
                     "final_session_share": cited(plain.get("final_session_share", ""), "[d2]"),
                     "vs_sp500": combined or cited(plain.get("vs_GSPC", ""), "[d2][d3]"),
                     "vs_nasdaq": combined or cited(plain.get("vs_IXIC", ""), "[d2][d4]"),
                     "volatility": cited(plain.get("volatility", ""), "[d2]"),
                     "drawdown": cited(plain.get("drawdown", ""), "[d2]"),
                     "volume_spike": cited(plain.get("volume_spike", ""), "[d2]"),
                     "dividend": cited(plain.get("dividend", ""), "[d2]"),
                     "prior_shock": cited(plain.get("prior_shock", ""),
                                          "[d2]" + (f"[{shock['ref']}]" if shock and shock["ref"] else ""))}
        if kind in sentences:
            if not sentences[kind]:
                raise ReportInputError(f"key finding '{kind}' is unavailable for this data; choose another kind")
            return sentences[kind]
        if kind == "data_limitation":
            if not limits:
                raise ReportInputError("there is no material data limitation to feature; choose another kind")
            return " ".join(limits)
        quote = _quote(ref, available[ref])
        if quote is None:
            raise ReportInputError(f"headline {ref} cannot be quoted safely; feature another headline or kind")
        return quote

    # Check every requested finding, including one displaced by a mandatory limitation.
    for kind, ref in requested_findings:
        featured_sentence(kind, ref, mandatory)

    # One day move in every sentence: the verified history's, which the findings also use.
    # The screener's figure only ranked the stock; a material gap is a data limitation.
    move = claims.path_facts(analysis)["final_session_return"]
    brief = "**In brief.** " + readable.headline([kind for kind, _ in findings], analysis, symbol=symbol,
                                                  trade_date=trade_iso, session_move=move, volume=volume, dividend=dividend,
                                                  index_day={})
    short_brief = "**In brief.** " + readable.headline([], {}, symbol=symbol, trade_date=trade_iso, session_move=move)

    def compose() -> list[str]:
        lead = list(dict.fromkeys(featured_sentence(kind, ref, mandatory) for kind, ref in findings))
        return [brief, "**Key findings.**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(lead, 1))]

    limit = settings.email.narrative_max_words
    # Required limitations survive the length budget; optional news can move to Sources.
    if word_count("\n\n".join(compose())) > limit:
        mandatory = _limits(state, analysis, compact=True, report_facts=report_facts)
    if word_count("\n\n".join(compose())) > limit:
        brief = short_brief
    while len(findings) > 1 and word_count("\n\n".join(compose())) > limit:
        removable = next((i for i in range(len(findings) - 1, -1, -1)
                          if findings[i][0] == 'news'), None)
        if removable is None:
            break
        findings.pop(removable)
    quoted = [ref for kind, ref in findings if kind == 'news']
    omitted = [hid for hid in ids if hid not in quoted]
    base = compose()
    # Omitted source titles remain in the separate headline/source list. No
    # truncation can remove a negation/condition from a quoted source statement.
    narrative = "\n\n".join(base)
    words = word_count(narrative)
    if words > limit:
        raise ReportInputError(f"report_length: computed report has {words} words; configured maximum is "
                               f"{limit}; no report was sent")
    lead = []
    for kind, ref in findings:
        text = featured_sentence(kind, ref, mandatory)
        if text not in [f['text'] for f in lead]:
            lead.append({"kind": kind, "ref": ref, "text": text})
    citations = list(dict.fromkeys(placeholders.CITATION.findall(narrative)))
    return EvidenceReport(narrative, citations, chosen.model_dump(), quoted, omitted, words, lead, source)

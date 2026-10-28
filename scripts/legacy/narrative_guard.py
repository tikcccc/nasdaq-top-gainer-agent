"""Historical free-prose report checker (superseded by evidence-v1; not used when sending).

Before evidence-v1 the model wrote the email body and this module checked it: template
structure, citations, computed price-path bindings, then Jev (or rules) for unsupported
facts, causality and advice. It is kept so the evaluations in docs/evaluation/history
can be reproduced; see scripts/eval_semantic_guard.py, eval_guard.py, eval_speculation.py
and eval_blind_reports.py.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from nasdaq_agent.config import Settings
from nasdaq_agent.infra.usage import BudgetExceeded
from nasdaq_agent.agent.evidence import analysis_problems
from nasdaq_agent.llm import jev
from nasdaq_agent.reporting import claims, placeholders, references
from nasdaq_agent.reporting.render import render_narrative

log = logging.getLogger(__name__)

REQUIRED = ("symbol", "pct_change", "trade_date")
ADVICE = re.compile(r"\b(you should (buy|sell)|we recommend|strong buy|buy now|sell now|price target of)\b", re.I)
UP_WORDS = re.compile(r"\b(up ?trend|upward trend|trend(ed|ing)? (higher|up)|rall(y|ied) (over|across) the)\b", re.I)
DOWN_WORDS = re.compile(r"\b(down ?trend|downward trend|trend(ed|ing)? (lower|down)|declin(ed|ing) over the)\b", re.I)


def word_count(narrative: str) -> int:
    """Words as written by the agent: each {{placeholder}} is one word; **markers** and [n1] citations are not."""
    text = placeholders.PLACEHOLDER.sub(" X ", narrative)
    text = placeholders.CITATION.sub(" ", text).replace("**", " ")
    return len(re.findall(r"[A-Za-z0-9][\w'’.%+$,-]*", text))


SPECULATION_QUESTION = (
    "This sentence asserts a cause, motive or funding link between events - for example that something happened "
    "because of, was driven by, was funded by, is tied to, or was done in order to achieve something else - and "
    "that link is not explicitly stated in the evidence. Merely reporting what a headline says, describing price "
    "moves or dates, restating the sentiment or catalyst classification, or describing a standard mechanism "
    "(such as a share price adjusting on an ex-dividend date) does not count. Explicit conditional future "
    "risks ('if a downgrade occurs, financing conditions could change') do not assert an actual causal event."
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
EXHAUSTIVE_NEWS = re.compile(
    r"\bonly\s+(?:company\s+)?(?:news|headline)(?:\s+item)?\s+in\s+circulation\b|"
    r"\b(?:only|all)\s+(?:company\s+)?news\s+published\b[^.!?]*\banywhere\b|"
    r"\bno\s+other\s+(?:company\s+)?news\s+(?:exists|was\s+published)\b", re.I)


def news_coverage_problems(text: str) -> list[str]:
    return ["unsupported exhaustive news coverage claim; describe only the fetched headlines"] if EXHAUSTIVE_NEWS.search(text) else []

UNSUPPORTED_FACT_QUESTION = (
    "The sentence asserts a factual company event or business activity/identity that the evidence does "
    "not establish. Plausible events absent from the evidence count as unsupported, even with 'likely' or "
    "'reportedly'. Check what occurred, to whom, when, and proposed versus completed actions. Pure "
    "descriptions of calculated prices, ranking labels, model estimates or explicit hypothetical future "
    "risks do not assert actual company events. Mere co-occurrence with a documented announcement does "
    "not assert causation. Use only the supplied evidence, not outside knowledge."
)


def factual_sentences(text: str) -> list[str]:
    """Check every section; an invented event in the overview is still a claim."""
    sentences = []
    for line in text.splitlines():
        body = re.sub(r"^\s*[-*]\s+", "", line.strip())
        body = re.sub(r"^\*\*[^*]+\*\*\s*", "", body)
        sentences.extend(s.strip() for s in _SENTENCE_END.split(body) if s.strip())
    return sentences


def claim_sentences(text: str) -> list[str]:
    """Causes can be asserted in any section, including an overview or risk."""
    return factual_sentences(text)


def speculation_scores(settings: Settings, facts: dict, sentences: list[str]) -> list[float]:
    """Jev's probability that each sentence makes an unsupported causal claim."""
    if not sentences:
        return []
    questions = {f"s{i}": jev.Noul(instructions=f"Sentence s{i}: {SPECULATION_QUESTION}") for i in range(1, len(sentences) + 1)}
    state = {"evidence": facts, "sentences": {f"s{i}": s for i, s in enumerate(sentences, 1)}}
    answer = jev.ask(settings, state, questions)
    return [float(answer.nouls[f"s{i}"].noul) for i in range(1, len(sentences) + 1)]


def _jev_speculation(settings: Settings, text: str, facts: dict) -> list[str]:
    sentences = claim_sentences(text)
    limit = settings.jev.speculation_threshold
    return [
        f"unsupported causal claim (p={p:.2f} > {limit}): '{s[:90]}' - state only links the sources say explicitly, "
        "or remove the sentence"
        for s, p in zip(sentences, speculation_scores(settings, facts, sentences)) if p > limit
    ]


CAUSAL_MARKERS = re.compile(
    r"\b(funded by|financed by|tied to|linked to|in order to|driven by|caused by|because of|as a result of)\b", re.I)


def _rule_speculation(text: str) -> list[str]:
    """Fallback when Jev is unavailable: flag explicit causal connectives in explanation sentences."""
    return [f"possible unsupported causal claim: '{s[:90]}' - state only links the sources say explicitly"
            for s in claim_sentences(text) if CAUSAL_MARKERS.search(s)]


def excluded_stock_mentions(state: dict, narrative: str) -> list[str]:
    """The excluded unfiltered #1 belongs in the method note only - never in the narrative."""
    raw = (state.get("raw_top") or {}).get("symbol")
    selected = (state.get("selected") or {}).get("symbol")
    issues = []
    mentions_symbol = raw and raw != selected and re.search(r"\b" + re.escape(raw) + r"\b", narrative, re.I)
    if mentions_symbol or "raw_top." in narrative or "[d7]" in narrative:
        issues.append("do not mention the excluded unfiltered #1 (its symbol, raw_top placeholders or [d7]); "
                      "the email's method note already explains it - remove it from the narrative")
    return issues


def _rule_consistency(text: str, facts: dict) -> list[str]:
    issues = []
    trend = facts.get("metrics.trend")
    if trend == "down" and UP_WORDS.search(text):
        issues.append("text describes an upward trend but metrics.trend is 'down'")
    if trend == "up" and DOWN_WORDS.search(text):
        issues.append("text describes a downward trend but metrics.trend is 'up'")
    if ADVICE.search(text):
        issues.append("text contains investment advice; describe, do not recommend")
    return issues


def evidence(state: dict, facts: dict, citations: set[str]) -> dict:
    """Everything the narrative may legitimately rely on: metrics, headlines, data checks, flags."""
    headlines = state.get("headlines") or {}
    selected = state.get("selected") or {}
    checks = state.get("checks") or {}
    sentiment_ids = set((state.get("sentiment") or {}).get("headline_ids") or [])
    daily = {}
    meta = (state.get("datasets") or {}).get(selected.get("dataset_id", ""))
    if meta:
        frame = pd.read_csv(meta["path"]).tail((selected.get("days") or 5) + 1)
        closes = frame["close"].tolist()
        daily = {
            str(d): {"close": round(c, 4), "change_vs_previous": placeholders.pct(c / p - 1) if p else None}
            for d, c, p in zip(frame["date"], closes, [None, *closes[:-1]])
        }
    return {
        "metrics_and_context": facts,
        "daily_closes_in_window": daily,
        "headlines": {h: {key: headlines[h].get(key) for key in ("title", "published", "publisher", "link", "symbol", "trade_date")}
                      for h in sorted(citations | sentiment_ids) if h in headlines},
        "news_coverage": "Only the fetched headlines, not an exhaustive record of company events or news in circulation.",
        "sentiment_assessment": {**(state.get("sentiment") or {}),
            "interpretation": "Model estimates of headline sentiment/catalyst/dilution risk, not observed event causality."},
        "verified_analysis": {key:value for key,value in (state.get("analysis") or {}).items() if key != "notes"},
        "derived_path_facts": claims.path_facts(claims.analysis_from_state(state)),
        "selected_stock": {key:selected.get(key) for key in ("symbol", "name", "price", "pct_change", "volume", "flags")},
        "data_checks": {
            "screener_vs_history": (checks.get("screener_vs_history") or {}).get("message"),
            "price_confirmation": (checks.get("price_confirmation") or {}).get("message"),
            "ranking": (state.get("ranking") or {}).get("message"),
            "corporate_actions_in_window": checks.get("corporate_actions") or "none",
            "quality_flags_of_selected_stock": selected.get("flags") or "none",
            "market_cap_usd": selected.get("market_cap"),
            "processing": checks.get("processing") or [],
        },
        "selection": state.get("selection") or {"label": "NASDAQ common-stock ranking"},
    }


def _jev_consistency(settings: Settings, text: str, facts: dict) -> list[str]:
    # One unit per line (each section and each risk bullet), matching how the email renders them;
    # splitting on blank lines would merge bullets into the paragraph above when the model omits a blank line.
    paragraphs = [line.strip() for line in text.splitlines()
                  if re.sub(r"\*\*[^*]+\*\*|\[[nd]\d+\]|\s+", "", line)]
    # Direction/support and unsupported factual assertions are separate judgements. In particular,
    # "not contradicted" must not allow a plausible event absent from this run's evidence.
    questions = {
        f"p{i}": jev.Noul(instructions=(
            f"Paragraph {i} of report_text is supported by the evidence: its statements about direction, "
            "trend, size of moves, benchmarks, news and data checks follow from the evidence without "
            "contradictions. Model assessments are estimates. Explicit hypothetical future risks and "
            "statements acknowledging insufficient evidence are acceptable."))
        for i in range(1, len(paragraphs) + 1)
    }
    questions["advice"] = jev.Noul(instructions="report_text tells the reader to buy, sell or hold the stock.")
    sentences = factual_sentences(text)
    questions.update({f"g{i}": jev.Noul(instructions=f"Sentence g{i}: {UNSUPPORTED_FACT_QUESTION}")
                      for i in range(1, len(sentences) + 1)})
    state = {"evidence": facts, "report_text": {f"paragraph_{i}": p for i, p in enumerate(paragraphs, 1)},
             "sentences": {f"g{i}": s for i, s in enumerate(sentences, 1)}}
    answer = jev.ask(settings, state, questions)
    issues = news_coverage_problems(text)
    threshold = settings.jev.consistency_threshold
    for i, paragraph in enumerate(paragraphs, 1):
        p = float(answer.nouls[f"p{i}"].noul)
        if p < threshold:
            issues.append(f"paragraph {i} ('{paragraph[:60]}...') contradicts or is not supported by the data "
                          f"(support {p:.2f} < {threshold}); correct or remove the claims in it the data does not show")
    advice = float(answer.nouls["advice"].noul)
    if advice > 0.7:
        issues.append(f"the text reads as investment advice (p={advice:.2f}); describe, do not recommend")
    for i, sentence in enumerate(sentences, 1):
        p = float(answer.nouls[f"g{i}"].noul)
        if p > settings.jev.unsupported_fact_threshold:
            issues.append(f"unsupported factual assertion (p={p:.2f}): '{sentence[:100]}' - "
                          "use only facts established by this run's evidence, or remove the assertion")
    return issues


def check_legacy_narrative(state: dict, args: dict, settings: Settings) -> tuple[list[str], dict]:
    """Historical free-prose evaluator only; never used by the sending graph/tool."""
    narrative = str(args.get("narrative") or "")
    citations = list(args.get("citations") or [])
    details: dict = {"method": None}
    if not state.get("analysis"):
        return [], details  # the tool itself reports the missing prerequisite

    selected = state.get("selected") or {}
    if selected.get("dataset_id"):
        issues = analysis_problems(state)
        if issues:
            return issues, details
    if state.get("selection") and "selection_label" not in placeholders.PLACEHOLDER.findall(narrative):
        return ["use {{selection_label}} to identify the ranking scope; do not claim an unfiltered #1"], details

    ctx = placeholders.build_context(state)
    facts = placeholders.flatten(ctx)
    used = set(placeholders.PLACEHOLDER.findall(narrative))
    template_problems = placeholders.problems(narrative, placeholders.flatten(placeholders.narrative_context(ctx)))
    problems = list(template_problems)
    problems += excluded_stock_mentions(state, narrative)
    problems += claims.problems(state, narrative)
    missing = [p for p in REQUIRED if p not in used]
    if missing:
        problems.append(f"the narrative must mention {['{{' + m + '}}' for m in missing]}")
    if len([p for p in used if p.startswith("metrics.")]) < 2 and not set(claims.PATH_KEYS).issubset(used):
        problems.append("use at least two {{metrics.*}} placeholders to describe the window")
    words = word_count(render_narrative(narrative, ctx)) if "path.sequence" in used and not template_problems else word_count(narrative)
    hi = settings.email.narrative_max_words
    if words > hi:
        problems.append(f"the narrative has {words} words; cut at least {words - hi} to stay within the {hi}-word maximum")

    refs = references.available(state)
    cited = set(placeholders.CITATION.findall(narrative)) | set(citations)
    unknown = sorted(c for c in cited if c not in refs)
    if unknown:
        problems.append(f"citations {unknown} do not match any source; use {sorted(refs)}")
    trade_date = (state.get("session") or {}).get("trade_date")
    for hid in (h for h in cited if h.startswith("n") and h in (state.get("headlines") or {})):
        h = state["headlines"][hid]
        if selected.get("analysis_key") and (h.get("symbol") != selected.get("symbol") or h.get("trade_date") != trade_date):
            problems.append(f"citation {hid} belongs to another symbol or trade date")
    sentiment = state.get("sentiment") or {}
    if selected.get("analysis_key") and any(p.startswith("sentiment.") for p in used):
        if sentiment.get("symbol") != selected.get("symbol") or sentiment.get("trade_date") != trade_date:
            problems.append("sentiment belongs to another symbol or trade date; run analyze_sentiment again")
    if not any(c.startswith("d") for c in cited):
        problems.append("cite the data behind price claims, e.g. [d2] for the stock's history and [d3]/[d4] for the benchmarks")
    bullets = [line.strip()[2:] for line in narrative.splitlines() if line.strip().startswith(("- ", "* "))]
    generic = [b[:50] for b in bullets if not placeholders.CITATION.search(b)]
    if generic:
        problems.append(f"risk bullets must be specific to this stock and cite evidence; drop or cite {generic}")
    if len(bullets) > 2:
        problems.append(f"{len(bullets)} risk bullets; keep at most two stock-specific ones")
    if problems:
        return problems, details

    text = render_narrative(narrative, ctx)
    semantic_text = render_narrative(claims.semantic_narrative(narrative), ctx)
    details["claim_binding"] = {"policy": "price-path-v1", "facts": claims.path_facts(claims.analysis_from_state(state))}
    try:
        if not jev.available(settings):
            raise RuntimeError("Jev disabled")
        facts_for_jev = evidence(state, facts, cited)
        problems += _jev_consistency(settings, semantic_text, facts_for_jev)
        problems += _jev_speculation(settings, semantic_text, facts_for_jev)
        details["method"] = "jev"
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - degrade to rules, never block on the checker itself
        log.warning("Jev consistency check unavailable: %s", exc)
        problems += _rule_consistency(semantic_text, facts)
        problems += _rule_speculation(semantic_text)
        details["method"] = "rules"
        if not settings.jev.allow_rule_fallback:
            problems.append("semantic_checker_unavailable: cannot establish factual support for a free-form narrative")
    details["rendered_preview"] = text[:400]
    return problems, details

import pytest

from scripts.legacy import narrative_guard as guard
from nasdaq_agent.reporting import placeholders
from nasdaq_agent.reporting.render import render_narrative
from nasdaq_agent.reporting.sparkline import sparkline

STATE = {
    "session": {"trade_date": "2026-10-23", "is_intraday": False},
    "selected": {"symbol": "AIOK", "name": "AI OKTO", "pct_change": 182.82, "price": 12.02, "volume": 41_847_594,
                 "days": 5, "flags": []},
    "raw_top": {"symbol": "ADNWW", "pct_change": 2800.0, "flags": ["derivative", "penny"]},
    "analysis": {"daily_returns": [-0.05, 0.24, -0.02, -0.12, 1.83], "avg_daily_return": 0.37,
                 "cumulative_return": 1.86, "trend": "up", "volatility_daily": 0.82, "volatility_annualized": 13.0,
                 "max_drawdown": -0.14, "benchmark": {"GSPC": -0.005, "IXIC": -0.007}, "excess_return": 1.87},
    "headlines": {"n1": {"id": "n1", "title": "Special dividend declared", "publisher": "GNW"}},
}


@pytest.mark.parametrize("text, found", [
    ("up 12.5% today", ["12.5%"]),
    ("closed at $3.50", ["$3.50"]),
    ("a ratio of 1.25", ["1.25"]),
    ("the S&P 500 over 5 days [n1] {{metrics.cumulative_return}}", []),
])
def test_raw_number_detection(text, found):
    assert placeholders.raw_numbers(text) == found


def test_context_formats_numbers():
    ctx = placeholders.build_context(STATE)
    assert ctx["pct_change"] == "+182.8%"
    assert ctx["metrics"]["cumulative_return"] == "+186.0%"
    assert ctx["metrics"]["gspc_return"] == "-0.50%"
    assert ctx["raw_top"]["symbol"] == "ADNWW"
    assert "sentiment" not in ctx  # only offered when sentiment exists


def test_render_rejects_unknown_placeholder_and_literals():
    ctx = placeholders.build_context(STATE)
    with pytest.raises(ValueError, match="unknown placeholders"):
        render_narrative("{{metrics.sharpe}} is high", ctx)
    with pytest.raises(ValueError, match="literal figures"):
        render_narrative("{{symbol}} rose 180%", ctx)
    with pytest.raises(ValueError, match="template statements"):
        render_narrative("{% for x in y %}{% endfor %}", ctx)
    assert render_narrative("{{symbol}} {{pct_change}}", ctx) == "AIOK +182.8%"


def test_readable_formats():
    assert placeholders.pct_units(2800.0) == "+2,800.0%"
    assert placeholders.pct_units(-12.191) == "-12.19%"
    assert placeholders.pct(13.08, 0, signed=False) == "1,308%"
    assert placeholders.compact(41_847_594) == "41.8M"
    assert placeholders.human_date("2026-10-23") == "Fri, Oct 23, 2026"
    assert placeholders.flag_text(["derivative", "penny"]) == "warrant/right/unit, sub-$1"


def test_sparkline():
    assert sparkline([1, 2, 3]) == "▁▄█"
    assert sparkline([2, 2]) == "▄▄"


def _check(narrative, citations=(), state=STATE, settings=None):
    from nasdaq_agent.config import Settings
    s = settings or Settings.model_validate({"jev": {"enabled": False}})
    s = s.model_copy(update={"jev": s.jev.model_copy(update={"allow_rule_fallback": True})})
    # These are structural/text unit fixtures; numeric snapshot integrity is tested
    # against real CSVs in test_recovery_and_state.py.
    if (state.get("selected") or {}).get("dataset_id") and not state["selected"].get("analysis_key"):
        state = {**state, "selected": {k:v for k,v in state["selected"].items() if k != "dataset_id"}}
    return guard.check_legacy_narrative(state, {"narrative": narrative, "citations": list(citations)}, s)[0]


# Neutral, number-free sentence (12 words) used to bring test narratives to a realistic length.
FILLER = "Source references identify the supplied price data and headlines in this report. "


def padded(text: str, words: int) -> str:
    """text plus filler until it has at least `words` words."""
    missing = max(0, words - guard.word_count(text))
    return text + "\n\n" + (FILLER * (missing // 12 + 1)).strip()


GOOD = padded("{{symbol}} rose {{pct_change}} on {{trade_date}}.\n\nOver {{days}} days it returned "
              "{{metrics.cumulative_return}} with an {{metrics.trend}} trend [d2][n1].", 170)


def test_guard_passes_good_narrative():
    assert _check(GOOD, ["n1"]) == []


def test_guard_structural_rejections():
    problems = _check("{{symbol}} jumped 180% on the day.")
    joined = " | ".join(problems)
    assert "literal figures" in joined and "{{pct_change}}" in joined and "metrics" in joined


def test_guard_rejects_unknown_citation():
    assert any("n9" in p for p in _check(GOOD + " See [n9].", ["n9"]))


def test_guard_rule_consistency_and_advice():
    down = {**STATE, "analysis": {**STATE["analysis"], "trend": "down"}}
    text = GOOD.replace("with an {{metrics.trend}} trend", "in a clear uptrend")
    assert any("upward trend" in p for p in _check(text + " {{metrics.excess_return}}", ["n1"], state=down))
    assert any("investment advice" in p for p in _check(GOOD + " We recommend investors buy now.", ["n1"]))


def test_word_count_rules():
    # placeholders count as one word; **markers** and [n1] citations are not words
    assert guard.word_count("**Why this stock.** {{symbol}} rose {{pct_change}} today [n1].") == 7
    assert guard.word_count("It fell -12.19% then rose $3.50, a 1-for-10 split.") == 9


def _words(n: int) -> str:
    base = "{{symbol}} rose {{pct_change}} on {{trade_date}}; over {{days}} days {{metrics.cumulative_return}} and {{metrics.trend}}. [d2]."
    extra = n - guard.word_count(base)
    return base + " " + " ".join(["steady"] * extra)


@pytest.mark.parametrize("n, ok", [(15, True), (149, True), (150, True), (250, True), (251, False)])
def test_guard_word_limit_boundaries(n, ok):
    text = _words(n)
    assert guard.word_count(text) == n
    problems = [p for p in _check(text) if "words" in p]
    assert (problems == []) is ok
    if not ok:
        assert "cut at least 1" in problems[0]


def test_guard_word_limit_is_configurable():
    from nasdaq_agent.config import Settings
    tight = Settings.model_validate({"jev": {"enabled": False}, "email": {"narrative_max_words": 20}})
    assert any("cut at least" in p for p in _check(_words(30), settings=tight))
    assert not any("words" in p for p in _check(_words(15), settings=tight))


REF_STATE = {
    **STATE,
    "sources": {"screener": "yahoo", "history": "yahoo"},
    "selected": {**STATE["selected"], "dataset_id": "ds_AIOK"},
    "datasets": {
        "ds_AIOK": {"symbol": "AIOK", "first_date": "2026-10-16", "last_date": "2026-10-23", "path": "x"},
        "ds_GSPC": {"symbol": "^GSPC", "first_date": "2026-10-16", "last_date": "2026-10-23", "path": "x"},
        "ds_IXIC": {"symbol": "QQQ", "proxy_for": "^IXIC", "first_date": "2026-10-16", "last_date": "2026-10-23", "path": "x"},
    },
    "headlines": {"n1": {"id": "n1", "title": "Special dividend declared", "publisher": "GNW",
                         "published": "2026-10-23T13:00:00+00:00", "link": "https://example.com/n1"}},
}


def test_references_point_to_checkable_pages():
    from nasdaq_agent.reporting import references
    refs = references.available(REF_STATE)
    assert refs["d1"].url == "https://finance.yahoo.com/markets/stocks/gainers/"
    assert refs["d2"].url == "https://finance.yahoo.com/quote/AIOK/history/"
    assert refs["d3"].url == "https://finance.yahoo.com/quote/%5EGSPC/history/"
    assert "qqq/historical" in refs["d4"].url and "proxy" in refs["d4"].title       # fallback source is cited honestly
    assert refs["d7"].url.endswith("/stocks/adnww")
    assert refs["n1"].url == "https://example.com/n1"
    assert not any("jev" in r.publisher.lower() for r in refs.values())          # models are not sources


def test_reference_numbering_follows_first_use():
    from nasdaq_agent.reporting import references
    refs = references.available(REF_STATE)
    numbered = references.number(refs, ["n1", "d2", "n1", "d9", "d1"])
    assert [(r.id, r.number) for r in numbered] == [("n1", 1), ("d2", 2), ("d1", 3)]


def test_guard_requires_valid_data_citations():
    no_data = GOOD.replace("[d2]", "")
    assert any("cite the data" in p for p in _check(no_data, state=REF_STATE))
    assert any("d9" in p for p in _check(GOOD + " See [d9].", state=REF_STATE))
    assert not any("cite" in p or "citations" in p for p in _check(GOOD, state=REF_STATE))


def test_rendered_citations_are_links():
    from nasdaq_agent.reporting import references
    from nasdaq_agent.reporting.render import _inline, _plain_citations
    refs = references.available(REF_STATE)
    references.number(refs, ["d2", "n1"])
    out = _inline("Closed higher [d2] on news [n1] and [d9].", refs)
    assert 'href="https://finance.yahoo.com/quote/AIOK/history/"' in out and ">[1]</a>" in out and ">[2]</a>" in out
    assert "d9" not in out                      # an id without a source is dropped, never shown unbacked
    assert _plain_citations("up [d2] [n1]", refs) == "up [1] [2]"


def test_guard_rejects_generic_risk_bullets():
    cited = GOOD + "\n- The special dividend comes off the price on its ex-date [n1]."
    assert not any("risk bullets" in p for p in _check(cited, state=REF_STATE))
    generic = GOOD + "\n- Small caps can reverse quickly."
    assert any("specific to this stock" in p for p in _check(generic, state=REF_STATE))
    too_many = GOOD + "".join(f"\n- Specific risk {i} [n1]." for i in range(3))
    assert any("at most two" in p for p in _check(too_many, state=REF_STATE))


def test_risk_bullets_are_split_from_paragraphs():
    from nasdaq_agent.reporting.render import split_narrative
    paragraphs, risks = split_narrative("**Why.** One.\n**How.** Two.\n- Risk A.\n- Risk B.")
    assert len(paragraphs) == 2 and len(risks) == 2 and "Risk A" in risks[0]


# ---------- fix 1: the excluded unfiltered #1 stays out of the narrative ----------

def test_guard_rejects_any_mention_of_the_excluded_stock():
    by_symbol = GOOD + " The raw leader ADNWW was a warrant."
    by_placeholder = GOOD + " The raw leader {{raw_top.symbol}} was excluded."
    by_citation = GOOD + " A warrant led the raw list [d7]."
    for text in (by_symbol, by_placeholder, by_citation):
        problems = _check(text, state=REF_STATE)
        assert any("excluded unfiltered #1" in p for p in problems), text[-40:]
    assert not any("excluded" in p for p in _check(GOOD, state=REF_STATE))


def test_raw_top_is_template_only():
    ctx = placeholders.build_context(STATE)
    assert "raw_top" in ctx                                    # the method note still uses it
    assert "raw_top" not in placeholders.narrative_context(ctx)
    with pytest.raises(ValueError, match="unknown placeholders"):
        render_narrative("{{symbol}} beat {{raw_top.symbol}}", ctx)


# ---------- fix 2: unsupported causal claims ----------

NEWS_PARA = ("**What's behind it.** The company declared a special dividend [n1]. "
             "The payout is funded by the vessel sale [n1].")


def test_claim_sentences_cover_explanations_and_risks():
    text = "**Why this stock.** It rose. Big day.\n" + NEWS_PARA + "\n- Ex-date ahead [n1]."
    assert guard.claim_sentences(text) == [
        "It rose.",
        "Big day.",
        "The company declared a special dividend [n1].",
        "The payout is funded by the vessel sale [n1].",
        "Ex-date ahead [n1].",
    ]


def test_jev_speculation_rejects_only_high_scoring_sentences(monkeypatch):
    from nasdaq_agent.config import Settings
    settings = Settings.model_validate({"jev": {"enabled": True, "api_key": "x", "speculation_threshold": 0.5}})
    monkeypatch.setattr(guard, "speculation_scores", lambda s, facts, sentences: [0.05, 0.9][: len(sentences)])
    problems = guard._jev_speculation(settings, NEWS_PARA, {})
    assert len(problems) == 1 and "funded by the vessel sale" in problems[0] and "p=0.90" in problems[0]


def test_rule_fallback_flags_explicit_causal_connectives():
    assert guard._rule_speculation(NEWS_PARA)                       # "funded by"
    assert not guard._rule_speculation("**What's behind it.** A special dividend was declared [n1].")


def test_speculation_eval_cases_are_well_formed():
    import json
    from pathlib import Path
    cases = json.loads((Path(__file__).parent.parent / "scripts" / "speculation_cases.json").read_text(encoding="utf-8"))["cases"]
    labels = {c["label"] for c in cases}
    assert labels <= {"reject", "pass", "out_of_scope"}
    assert sum(c["label"] == "reject" for c in cases) >= 5 and sum(c["label"] == "pass" for c in cases) >= 5


def test_grounding_covers_every_section_and_each_sentence():
    text = "**Why this stock.** A company event. Another event.\n**How it traded.** Prices rose.\n- A risk [n1]."
    assert guard.factual_sentences(text) == ["A company event.", "Another event.", "Prices rose.", "A risk [n1]."]


def test_absence_of_contradiction_does_not_bypass_grounding(monkeypatch):
    from types import SimpleNamespace
    from nasdaq_agent.config import Settings
    settings = Settings.model_validate({"jev": {"enabled": True, "api_key": "x"}})
    captured = {}
    def ask(settings, state, questions):
        captured.update(state)
        probabilities={name: .99 if name.startswith('p') else .9 if name=='g2' else .01 for name in questions}
        return SimpleNamespace(nouls={name:SimpleNamespace(noul=value) for name,value in probabilities.items()})
    monkeypatch.setattr(guard.jev,"ask",ask)
    issues=guard._jev_consistency(settings,"**Overview.** Supported event. Invented event.",{"headlines":{"n1":"Supported event."}})
    assert len(issues)==1 and "Invented event" in issues[0]
    assert captured["sentences"]["g2"]=="Invented event."


def test_semantic_evidence_retains_news_metadata_and_estimate_scope():
    headline={**STATE["headlines"]["n1"],"published":"2026-10-23T13:00:00+00:00","link":"https://example.invalid/release"}
    state={**STATE,"headlines":{"n1":headline},"analysis":{**STATE["analysis"],"notes":"untrusted interpretation"}}
    evidence=guard.evidence(state,{}, {"n1"})
    assert evidence["headlines"]["n1"]["published"]==headline["published"]
    assert evidence["headlines"]["n1"]["link"]==headline["link"]
    assert "notes" not in evidence["verified_analysis"]
    assert "not an exhaustive" in evidence["news_coverage"]
    assert "Model estimates" in evidence["sentiment_assessment"]["interpretation"]


def test_global_news_claim_is_distinct_from_fetched_subset():
    assert guard.news_coverage_problems("The only company news in circulation was an earnings release.")
    assert guard.news_coverage_problems("This is the only news published about the company anywhere.")
    assert not guard.news_coverage_problems("This is the only returned headline.")

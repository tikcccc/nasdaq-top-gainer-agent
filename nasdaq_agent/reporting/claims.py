"""Deterministic price-path statements and conservative boundaries for free prose.

Only the validated analysis supplies prices. Attribution uses additive changes
in an index starting at one, not sums of percentage returns (which compound).
The language policy is a reporting contract, not a proof for arbitrary English.
"""
from __future__ import annotations

import math
import re

PATH_KEYS = ("path.sequence", "path.contribution", "path.benchmarks")
TOKEN = re.compile(r"{{\s*([\w.]+)\s*}}")
CITE = re.compile(r"\[([nd]\d+)\]")
TOL = 1e-9


def analysis_from_state(state: dict) -> dict:
    """Use exact snapshot closes, rather than tolerated/rounded generated returns."""
    selected = state.get("selected") or {}
    if not selected.get("analysis_key"):
        return state.get("analysis") or {}  # unversioned structural fixtures only
    try:
        import pandas as pd
        from nasdaq_agent.data import metrics
        datasets=state.get("datasets") or {}
        stock=pd.read_csv(datasets[selected["dataset_id"]]["path"])
        benches={ds[3:]:pd.read_csv(meta["path"]) for ds,meta in datasets.items()
                 if meta.get("proxy_for") or meta["symbol"].startswith('^')}
        return metrics.compute(stock,benches,selected['days'],selected['symbol'])
    except (KeyError,OSError,ValueError):
        return {}  # no trustworthy bound sentences; snapshot verification reports the failure


def path_facts(analysis: dict) -> dict:
    returns = analysis.get("daily_returns") or []
    if not returns or any(not isinstance(r, (int, float)) or not math.isfinite(r) or r <= -1 for r in returns):
        return {}
    indexed = [1.0]
    for r in returns:
        indexed.append(indexed[-1] * (1 + r))
    signs = ["up" if r > TOL else "down" if r < -TOL else "flat" for r in returns]
    total, prior, final = indexed[-1] - 1, indexed[-2] - 1, indexed[-1] - indexed[-2]
    share = final / total if abs(total) > TOL else None
    return {"session_directions": signs, "up_sessions": signs.count("up"),
            "down_sessions": signs.count("down"), "flat_sessions": signs.count("flat"),
            "window_return": total, "pre_final_return": prior, "final_session_return": returns[-1],
            "final_index_change": final, "final_share_of_net_change": share,
            "final_accounts_for_all_net_change": abs(prior) <= TOL and abs(total) > TOL,
            "final_accounts_for_majority_net_change": share is not None and .5 < share < 1 - TOL}


def text(analysis: dict) -> dict[str, str]:
    facts = path_facts(analysis)
    if not facts:
        return {}
    pct = lambda value: f"{value * 100:+.2f}%"
    directions = ", ".join(facts["session_directions"])
    sequence = (f"From oldest to newest, the session directions were {directions}. "
                f"The cumulative price return was {pct(facts['window_return'])}.")
    if all(isinstance(analysis.get(k), (int, float)) and math.isfinite(analysis[k])
           for k in ("volatility_daily", "max_drawdown")):
        sequence += (f" Daily sample volatility was {analysis['volatility_daily'] * 100:.2f}% "
                     f"and maximum drawdown was {pct(analysis['max_drawdown'])}.")
    total, prior, final = facts["window_return"], facts["pre_final_return"], facts["final_index_change"]
    contribution = (f"The earlier sessions compounded to {pct(prior)}; "
                    f"the final session returned {pct(facts['final_session_return'])}.")
    if abs(total) <= TOL:
        contribution += " The net window change was approximately zero, so a percentage share of it is not meaningful."
    elif facts["final_accounts_for_all_net_change"]:
        contribution += " The final session accounts for all of the net window change; the earlier sessions netted to zero."
    elif facts["final_accounts_for_majority_net_change"]:
        contribution += (f" It contributed most ({facts['final_share_of_net_change'] * 100:.1f}%) "
                         "of the net window change, measured in indexed price units.")
    elif final * total < -TOL:
        contribution += " The final session offset part of the earlier net move."
    elif abs(final) <= TOL:
        contribution += " The final session added no material change; the earlier sessions account for the net move."
    elif prior * final < -TOL:
        contribution += " The final session reversed the direction of the earlier net move."
    else:
        contribution += " Both the earlier sessions and the final session contributed to the net window change."
    comparisons = []
    for key, name in (("GSPC", "S&P 500"), ("IXIC", "NASDAQ Composite")):
        value = (analysis.get("benchmark") or {}).get(key)
        if value is None:
            continue
        gap = total - value
        direction = "above" if gap > TOL else "below" if gap < -TOL else "equal to"
        comparisons.append(f"{name} returned {pct(value)}; the stock's window return was {direction} that benchmark")
    out = {"sequence": sequence, "contribution": contribution,
           "benchmarks": ". ".join(comparisons) + "." if comparisons else "No matching benchmark window was available for comparison."}
    # Single-fact sentences, so the agent's key findings can feature one without repeating it.
    out["window"] = (f"From oldest to newest, the session directions were {directions}. "
                     f"The cumulative price return was {pct(facts['window_return'])}.")
    if "Daily sample volatility" in sequence:
        out["volatility"] = f"Daily sample volatility was {analysis['volatility_daily'] * 100:.2f}%."
        out["drawdown"] = f"Maximum drawdown over the window was {pct(analysis['max_drawdown'])}."
    for key, name in (("GSPC", "S&P 500"), ("IXIC", "NASDAQ Composite")):
        line = next((c for c in comparisons if c.startswith(name)), None)
        if line:
            out[f"benchmark_{key}"] = line + "."
    return out


_SECTION = re.compile(r"^\s*\*\*How it traded\.?\*\*\s*(.*?)(?=^\s*\*\*|\Z)", re.I | re.M | re.S)
_PRICE = re.compile(r"\b(stock|shares?|prices?|clos(?:e[sd]?|ing)|sessions?|days?|gains?|loss(?:es)?|returns?|advance|rall(?:y|ied)|decline|window|week|path|trend)\b", re.I)
_QUANTIFIER = re.compile(r"\b(all|entire|whole|sole|only|most|majority|bulk|predominant\w*|each|every|consecutive|consisten\w*|stead\w*|uninterrupted|alternat\w*|throughout|day.by.day|day.after.day|single|concentrat\w*|back.loaded|front.loaded|progressively)\b", re.I)
_COMPARATIVE = re.compile(r"\b(sharpest|biggest|largest|smallest|strongest|weakest|highest|lowest|best|worst|unprecedented|historic(?:ally)?|all.time|record|record.breaking|record.setting|mega.cap|large.cap|mid.cap|small.cap)\b", re.I)
_BENCH = re.compile(r"\b(outperform\w*|underperform\w*|beat|beaten|lagged|trailing|ahead of)\b.*\b(benchmark|S&P|NASDAQ|market|index|indices)\b", re.I)
_CAUSE = re.compile(r"\b(because(?: of)?|driven by|caused by|funded by|financed by|fuel(?:l)?ed by|spurred by|triggered by|powered by|thanks to|in response to|in order to|led to|resulted in|explains?|attributed to|stemmed from|due to|on the back of|(?:earnings|news|results)[- ]driven)\b|\b(?:rose|rallied|jumped|gained|fell|slid|dropped|declined)\s+on\s+", re.I)
_NEGATION = re.compile(r"\b(cannot|can't|does not|do not|did not|not establish|not prove|no evidence|not known|not attributed)\b", re.I)
_ABSENCE = re.compile(r"\b(no|without|absence of|lack of)\b[^.!?]*\b(news|headlines?|driver|catalyst|announcement|disclosed)\b", re.I)
_GLOBAL = re.compile(r"\b(public(?:ly)?|anywhere|circulation|exists?|all sources)\b", re.I)
_FETCHED = re.compile(r"\b(fetched|returned|retrieved|supplied|provided)\b", re.I)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def _without_supported_quotes(sentence: str, state: dict) -> str:
    """An exact attributed headline quote is a source claim, not our causal conclusion."""
    if not re.search(r"\b(headline|source|reported|reports|according to|stated|says)\b", sentence, re.I):
        return sentence
    cited = {m for m in CITE.findall(sentence) if m.startswith("n")}
    titles = [(state.get("headlines") or {}).get(h, {}).get("title", "").casefold() for h in cited]
    def remove(match):
        quote = match.group(1).casefold().strip()
        return "SOURCE_QUOTE" if quote and any(quote == title for title in titles) else match.group(0)
    return re.sub(r'["“]([^"”]+)["”]', remove, sentence)


def problems(state: dict, narrative: str, *, require_bindings: bool | None = None) -> list[str]:
    """Protect calculated path language even when the semantic service is unavailable.

    Legacy unversioned structural fixtures need not contain a full report. Real
    analyses always have analysis_key and must use the protected paragraph.
    """
    required = bool((state.get("selected") or {}).get("analysis_key")) if require_bindings is None else require_bindings
    issues = []
    used = set(TOKEN.findall(narrative))
    if required or set(PATH_KEYS) & used:
        sections = list(_SECTION.finditer(narrative))
        section = sections[0] if len(sections)==1 else None
        expected = " ".join("{{" + key + "}}" for key in PATH_KEYS)
        body = CITE.sub("", section.group(1)).strip() if section else ""
        body = re.sub(r"\s+", " ", body)
        body = re.sub(r"{{\s*([\w.]+)\s*}}", r"{{\1}}", body)
        if body != expected or any(TOKEN.findall(narrative).count(key)!=1 for key in PATH_KEYS):
            issues.append("bound_price_path_required: How it traded must contain only " + expected + " and data citations")
        if "news.coverage" not in used:
            issues.append("bound_news_scope_required: use {{news.coverage}} to describe the fetched news scope")
    # Code-produced strings are removed before checking raw model prose. Never
    # exempt the surrounding sentence just because it contains one safe token.
    raw = TOKEN.sub("BOUND_VALUE", narrative)
    for sentence in _SENTENCES.split(raw):
        if not sentence.strip():
            continue
        scope = _without_supported_quotes(sentence, state)
        price = bool(_PRICE.search(scope))
        if price and _QUANTIFIER.search(scope):
            issues.append("unbound_path_claim: replace price-path quantifiers with {{path.sequence}} or {{path.contribution}}")
        if (price and _COMPARATIVE.search(scope)) or _BENCH.search(scope):
            issues.append("unsupported_comparison: use {{path.benchmarks}} for the matched window; remove historical superlatives and unsupported size classes")
        if _ABSENCE.search(scope) and (_GLOBAL.search(scope) or not _FETCHED.search(scope)):
            issues.append("unbounded_news_absence: an empty fetched set does not establish absence of public news or catalysts; use {{news.coverage}}")
        for clause in re.split(r";|\b(?:but|although|though|however|yet|nevertheless)\b", scope, flags=re.I):
            if not _CAUSE.search(clause):
                continue
            if _NEGATION.search(clause):
                continue
            if re.search(r"\bif\b", clause, re.I) and re.search(r"\b(could|would|might|may)\b", clause, re.I):
                continue
            issues.append("unbound_causal_claim: do not infer event causation or funding links; an attributed exact headline quote with its news citation is allowed")
    return list(dict.fromkeys(issues))


def semantic_narrative(narrative: str) -> str:
    """Do not ask a probabilistic service to re-judge code-rendered mathematics."""
    if set(PATH_KEYS).issubset(TOKEN.findall(narrative)):
        expected = " ".join("{{" + key + "}}" for key in PATH_KEYS)
        def remove_safe(match):
            body = re.sub(r"\s+", " ", CITE.sub("", match.group(1))).strip()
            body = TOKEN.sub(lambda m:"{{"+m.group(1)+"}}",body)
            return "" if body==expected else match.group(0)
        narrative = _SECTION.sub(remove_safe, narrative)
    return TOKEN.sub(lambda m: "" if m.group(1).startswith("path.") or m.group(1)=="news.coverage" else m.group(0), narrative)

"""Turn validated state + the agent's narrative into the email (HTML + text + inline chart)."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from jinja2.exceptions import TemplateError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from nasdaq_agent.reporting import announcements, placeholders, references, readable
from nasdaq_agent.reporting.facts import ReportFacts, build_report_facts
from nasdaq_agent.reporting.dates import news_date
from nasdaq_agent.reporting.version import renderer_identity
from nasdaq_agent.reporting.chart import normalised_chart
from nasdaq_agent.reporting.sparkline import sparkline
from nasdaq_agent.reporting.svg_chart import svg_chart

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"
UP, DOWN, INK, MUTED = "#15803d", "#b91c1c", "#0f172a", "#64748b"
METHOD_NAMES = {"jev": "Jev", "llm": "LLM"}  # sentiment scorer (a model, not a data source)

_narrative_env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
_page_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))


@dataclass
class Report:
    subject: str
    html: str
    text: str
    images: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)      # run-log diagnostics, never in the email
    facts: dict = field(default_factory=dict)
    renderer: dict = field(default_factory=dict)


def render_narrative(text: str, ctx: dict) -> str:
    """Fill placeholders. Raises ValueError with an actionable message."""
    ctx = placeholders.narrative_context(ctx)
    issues = placeholders.problems(text, placeholders.flatten(ctx))
    if issues:
        raise ValueError("; ".join(issues))
    try:
        return _narrative_env.from_string(text).render(**ctx)
    except (UndefinedError, TemplateError) as exc:
        raise ValueError(f"template error: {exc}") from exc


def cite_link(ref: references.Reference) -> str:
    label = html.escape(ref.publisher + ": " + ref.title)
    if not ref.url:
        return f'<span title="{label}" style="color:#2563eb;font-weight:600;">[{ref.number}]</span>'
    return (f'<a href="{html.escape(ref.url)}" title="{label}" '
            f'style="color:#2563eb;text-decoration:none;font-weight:600;">[{ref.number}]</a>')


def _inline(text: str, refs: dict[str, references.Reference] | None = None) -> str:
    """Escape, then **bold** and citations -> numbered links (inline styles: email clients drop classes)."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", rf'<strong style="color:{INK};">\1</strong>', out)

    def cite(match: re.Match) -> str:
        ref = (refs or {}).get(match.group(1))
        if not ref or not ref.number:
            return ""  # never show a citation we cannot back with a link
        return f'<sup style="font-size:10px;">{cite_link(ref)}</sup>'

    return placeholders.CITATION.sub(cite, out)


def _plain_citations(text: str, refs: dict[str, references.Reference]) -> str:
    def cite(match: re.Match) -> str:
        ref = refs.get(match.group(1))
        return f"[{ref.number}]" if ref and ref.number else ""

    return placeholders.CITATION.sub(cite, text)


def narrative_blocks(text: str, refs: dict[str, references.Reference] | None = None) -> list[dict]:
    """Paragraphs and bullet lists of the narrative, ready for the template."""
    blocks: list[dict] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        bullets = [line[2:] for line in lines if line.startswith(("- ", "* "))]
        prose = [line for line in lines if not line.startswith(("- ", "* "))]
        # One paragraph per line: models often put the "**Why this stock.**" sections on consecutive lines.
        blocks += [{"kind": "p", "html": _inline(p, refs)} for p in prose]
        if bullets:
            blocks.append({"kind": "ul", "bullets": [_inline(b, refs) for b in bullets]})
    return blocks


def split_narrative(text: str, refs: dict[str, references.Reference] | None = None) -> tuple[list[dict], list[str]]:
    """(paragraphs, risk bullets). Risks are rendered in a low-key line near the end of the email."""
    paragraphs, risks = [], []
    for block in narrative_blocks(text, refs):
        if block["kind"] == "p":
            paragraphs.append(block)
        else:
            risks += block["bullets"]
    return paragraphs, risks


def _benchmark_changes(state: dict, facts: ReportFacts) -> dict[str, dict[str, float]]:
    """{'S&P 500': {date: day change}, 'NASDAQ': {...}} for the daily table."""
    out = {}
    for ds, meta in (state.get("datasets") or {}).items():
        key = meta.get("proxy_for") or meta["symbol"]
        label = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ"}.get(key)
        if not label:
            continue
        frame = facts.frames.get(ds)
        if frame is None:
            continue
        out[label] = {str(frame["date"].iloc[i]): float(frame["close"].iloc[i] / frame["close"].iloc[i - 1] - 1)
                      for i in range(1, len(frame))}
    return out


def _daily_rows(facts: ReportFacts, state: dict) -> list[dict]:
    """One row per session: close, change, bar, benchmark changes and a hover title with everything."""
    frame = facts.stock
    if frame is None:
        return []
    bench = _benchmark_changes(state, facts)
    start = float(frame["close"].iloc[0])
    changes = [float(frame["close"].iloc[i] / frame["close"].iloc[i - 1] - 1) for i in range(1, len(frame))]
    biggest = max((abs(c) for c in changes), default=0) or 1
    rows = []
    for i, change in enumerate(changes, 1):
        iso = str(frame["date"].iloc[i])
        close = float(frame["close"].iloc[i])
        b = {k: v.get(iso) for k, v in bench.items()}
        title = (f"{iso} · close {placeholders.money(close)} · day {placeholders.pct(change)} · "
                 f"since start {placeholders.pct(close / start - 1)} · "
                 + " · ".join(f"{k} {placeholders.pct(v)}" for k, v in b.items() if v is not None))
        rows.append({
            "date": date.fromisoformat(iso).strftime("%a %b %d").replace(" 0", " "),
            "close": placeholders.money(close),
            "change": placeholders.pct(change),
            "color": UP if change >= 0 else DOWN,
            "bar": max(3, round(abs(change) / biggest * 100)),
            "delay": f"{0.15 * i:.2f}s",
            "spx": placeholders.pct(b.get("S&P 500")) if b.get("S&P 500") is not None else "–",
            "ndx": placeholders.pct(b.get("NASDAQ")) if b.get("NASDAQ") is not None else "–",
            "title": title,
        })
    return rows


def _kpis(ctx: dict, facts: ReportFacts) -> list[dict]:
    """The few numbers a reader takes away; the rest are in the text and the daily table."""
    analysis = facts.analysis
    if not analysis:
        return [{"label": "Day move", "value": ctx["pct_change"], "color": DOWN if (facts.day_return or 0) < 0 else UP}]
    m = ctx["metrics"]
    tone = lambda x: UP if (x or 0) >= 0 else DOWN  # noqa: E731
    tiles = [
        {"label": "Day move", "value": ctx["pct_change"], "color": tone(facts.day_return)},
        {"label": f"{ctx['days']}-day return", "value": m["cumulative_return"], "color": tone(analysis.get("cumulative_return"))},
    ]
    if analysis.get('excess_return') is not None:
        tiles.append({"label": f"{ctx['days']}-day excess vs S&P 500", "value": m["excess_return_pp"], "color": tone(analysis['excess_return'])})
    volume = facts.volume
    if volume is not None:
        tiles.append({"label": f"Volume / prior {volume['sessions']}-session avg",
                      "value": f"{readable.times(volume['ratio'])}×", "color": INK})
    return tiles


def _news(state: dict, citations: list[str]) -> list[dict]:
    headlines = state.get("headlines") or {}
    sentiment = state.get("sentiment") or {}
    candidates = [c for c in citations if c in headlines] + list(sentiment.get("headline_ids") or [])
    ids = list(dict.fromkeys(candidates))  # de-duplicate, keep first-cited order
    out = []
    for hid in ids[:5]:
        h = headlines.get(hid)
        if not h:
            continue
        selected = state.get("selected") or {}
        if selected.get("analysis_key") and (h.get("symbol") != selected.get("symbol") or
                h.get("trade_date") != (state.get("session") or {}).get("trade_date")):
            continue
        published = news_date(h.get('published'))
        when = readable.short_date(published) if published else ''
        out.append({**h, "when": when})
    return out


def _sentiment_view(state: dict) -> dict | None:
    s = state.get("sentiment") or {}
    if not s or s.get("method") == "none":
        return None
    selected = state.get("selected") or {}
    if selected.get("analysis_key") and (s.get("symbol") != selected.get("symbol") or
            s.get("trade_date") != (state.get("session") or {}).get("trade_date")):
        return None
    score = s.get("score") or 0
    color = UP if score >= 2.5 else DOWN if score <= 1.5 else MUTED
    return {
        "label": s.get("label", "n/a"),
        "score": f"{score:.1f}",
        "confidence": f"{(s.get('confidence') or 0) * 100:.0f}%",
        "catalyst": placeholders.CATALYST_LABELS.get(s.get("catalyst") or "", s.get("catalyst") or "n/a"),
        "dilution": f"{(s.get('dilution_risk') or 0) * 100:.0f}%",
        "dilution_high": (s.get("dilution_risk") or 0) > 0.5,
        "uncertain": s.get("uncertain"),
        "method": METHOD_NAMES.get(s.get("method"), s.get("method")),
        "color": color,
        "meter": round(score / 4 * 100),
    }


def _notes(state: dict, fallback_reason: str | None) -> list[str]:
    """Diagnostics for the run log, not the email: what matters to a reader is in the report's limitations."""
    checks = state.get("checks") or {}
    cross = checks.get("screener_vs_history") or checks.get("cross_source") or {}
    confirmation = checks.get("price_confirmation") or {}
    ranking = state.get("ranking") or {}
    notes = []
    if cross.get("status") in {"warn", "conflict"}:
        notes.append(cross["message"])
    if confirmation.get("status") in {"single_source", "conflict"}:
        notes.append(confirmation["message"][:1].upper() + confirmation["message"][1:] + ".")
    if ranking.get("status") == "uncertain":
        notes.append(f"Ranking: {ranking['message']}.")
    notes += [f"Corporate action in window: {a}" for a in checks.get("corporate_actions") or []]
    if (state.get("session") or {}).get("is_intraday"):
        notes.append("Market was open when this ran: figures are intraday and not final.")
    if fallback_reason and not state.get("analysis"):
        notes.append("Multi-day analysis is unavailable for this report.")
    flags = (state.get("selected") or {}).get("flags") or []
    if (state.get('selected') or {}).get('market_cap') is not None:
        notes.append('Screener market cap is unverified; omitted without inferring staleness from turnover.')
    if flags:
        notes.append("Flags: " + ", ".join(placeholders.FLAG_LABELS.get(f, f) for f in flags))
    raw_top = state.get('raw_top') or {}
    if raw_top and raw_top.get('symbol') != (state.get('selected') or {}).get('symbol'):
        notes.append(f"Unfiltered leader: {raw_top['symbol']} ({placeholders.pct_units(raw_top.get('pct_change'))}); outside the selected scope.")
    for item in (state.get('selection') or {}).get('split_excluded') or []:
        notes.append(f"Split-adjusted exclusion: {item['symbol']}, screener {item['screener_pct']:+.2f}%, "
                     f"adjusted {item['adjusted_pct']:+.2f}%.")
    return notes


def _subject(ctx: dict, state: dict, fallback_reason: str | None, run_id: str) -> str:
    trade = (state.get("session") or {}).get("trade_date")
    short = date.fromisoformat(trade).strftime("%b %d").replace(" 0", " ") if trade else ""
    if ctx["symbol"] == "n/a":
        return f"NASDAQ top-gainer report could not be completed{f' ({short})' if short else ''}"
    subject = f"{ctx['symbol']} {ctx['pct_change']} · {ctx['selection_label']}, {short}"
    if state.get("analysis"):
        subject += f" · {ctx['days']}-day {ctx['metrics']['cumulative_return']}"
    if fallback_reason and not state.get("analysis"):
        subject += " (partial report)"
    return subject


def _preheader(ctx: dict, state: dict) -> str:
    if ctx["symbol"] == "n/a":
        return "The report could not be completed. Please try again later."
    verb = 'last traded' if (state.get('session') or {}).get('is_intraday') else 'closed'
    parts = [f"{ctx['company'] or ctx['symbol']} {verb} at {ctx['price']} on {ctx['volume']} shares."]
    if "metrics" in ctx:
        parts.append(f"{ctx['days']}-day {ctx['metrics']['cumulative_return']} vs S&P 500 {ctx['metrics']['gspc_return']}.")
    if "sentiment" in ctx:
        parts.append(f"Headline sentiment estimate: {ctx['sentiment']['label']}.")
    return " ".join(parts)


def _chart(state: dict, run_dir: Path, facts: ReportFacts) -> tuple[Path | None, str | None]:
    """(static PNG for clients without SVG, animated inline SVG for the rest)."""
    selected = state.get("selected") or {}
    datasets = state.get("datasets") or {}
    if facts.stock is None or len(facts.stock) < 2:
        return None, None
    days = selected.get("days") or 5
    png = normalised_chart(datasets, selected["dataset_id"], run_dir / "chart_email.png", days, frames=facts.frames)
    return png, svg_chart(datasets, selected["dataset_id"], days, frames=facts.frames)


def _method(facts: ReportFacts, state: dict) -> list[str]:
    """Two reader-facing lines, plus a short action note only when needed.

    Capture timestamps, raw leaders and split diagnostics remain in the run record.
    """
    if facts.symbol == 'n/a':
        return []
    filters = (state.get('selection') or {}).get('filters') or {}
    scope = ['NASDAQ common stocks' if filters.get('exclude_derivatives', True) else 'NASDAQ listings']
    if filters.get('min_price'):
        scope.append(f"price ≥${filters['min_price']:g}")
    if filters.get('min_volume'):
        scope.append(f"volume ≥{placeholders.compact(filters['min_volume']).replace('.0K', 'K').replace('.0M', 'M')} shares")
    if filters.get('min_market_cap'):
        scope.append(f"market cap ≥${placeholders.compact(filters['min_market_cap'])}")
    lines = ['Screen: ' + ' · '.join(scope) + ' [d1].']
    if facts.stock is None:
        return lines + ["Multi-session price history is unavailable."]
    frame = facts.stock
    start, end = (date.fromisoformat(str(frame['date'].iloc[i])) for i in (0, -1))
    span = (f"{start.strftime('%b')} {start.day}–{end.day}, {end.year}" if (start.year, start.month) == (end.year, end.month) else
            f"{start.strftime('%b')} {start.day}–{end.strftime('%b')} {end.day}, {end.year}" if start.year == end.year else
            f"{start.isoformat()}–{end.isoformat()}")
    providers = [facts.source]
    data_refs = '[d2]'
    for index, ref in (('^GSPC', 'd3'), ('^IXIC', 'd4')):
        for ds, meta in (state.get('datasets') or {}).items():
            if (meta.get('proxy_for') or meta.get('symbol')) == index and ds in facts.frames:
                data_refs += f'[{ref}]'
                vendor = meta.get('source') or (state.get('sources') or {}).get(f'benchmark {index}')
                if vendor and vendor not in providers:
                    providers.append(vendor)
                break
    names = {'yahoo': 'Yahoo Finance', 'nasdaq': 'Nasdaq.com'}
    provider = ' / '.join(names.get(p, p) for p in providers)
    intraday = 'intraday · ' if facts.is_intraday else ''
    lines.append(f"Data: {provider} · {span} · {intraday}{len(frame) - 1}-session price returns {data_refs}.")
    splits = (state.get('checks') or {}).get('splits') or []
    if splits:
        lines.append('Volume comparison withheld across the split [d2].' if not facts.volume_comparable else
                     'Earlier prices and volumes restated for the split [d2].')
    return lines


def build_report(state: dict, narrative: str, *, run_dir: Path, run_info: dict, citations: list[str] | None = None,
                 fallback_reason: str | None = None) -> Report:
    facts = build_report_facts(state)
    ctx = placeholders.build_context(state, report_facts=facts)
    frame = facts.stock
    rows = _daily_rows(facts, state)
    cited = placeholders.CITATION.findall(narrative)
    news = _news(state, [c for c in cited if c.startswith("n")] + list(citations or []))
    chart, svg = _chart(state, run_dir, facts)
    images = {"chart": chart} if chart else {}
    subject = _subject(ctx, state, fallback_reason, run_info.get("run_id", ""))

    # Academic-style numbering: narrative citations first, then what the tables/chart/news rely on.
    refs = references.available(state)
    method = _method(facts, state)
    visible_news = [n for n in news if n['title'] not in narrative]
    order = cited + placeholders.CITATION.findall('\n'.join(method)) + [n["id"] for n in visible_news]
    sources = references.number(refs, order)
    for n in news:
        n["ref"] = refs.get(n["id"])
    paragraphs, risks = split_narrative(narrative, refs)
    summary = paragraphs.pop(0) if narrative.startswith('**In brief.**') and paragraphs else None
    notes = _notes(state, fallback_reason)
    renderer = renderer_identity()
    for note in notes:
        log.info("report note (log only): %s", note)

    page = _page_env.get_template("email.html.j2").render(
        subject=subject, preheader=_preheader(ctx, state), ctx=ctx, kpis=_kpis(ctx, facts),
        renderer=renderer,
        method=[_inline(line, refs) for line in method], is_intraday=facts.is_intraday,
        summary=summary, narrative=paragraphs, risks=risks, rows=rows,
        news=visible_news, sentiment=_sentiment_view(state),
        chart=chart, svg=svg,
        trade_date_long=placeholders.human_date((state.get("session") or {}).get("trade_date")),
        sources=sources, cite=cite_link,
        colors={"up": UP, "down": DOWN, "ink": INK, "muted": MUTED},
    )

    text_lines = [subject, "", _plain_citations(narrative, refs), ""]
    if "metrics" in ctx:
        m = ctx["metrics"]
        text_lines += [f"Day move {ctx['pct_change']} | {ctx['days']}-day return {m['cumulative_return']} | "
                       f"{ctx['days']}-day excess vs S&P 500 {m['excess_return_pp']}", ""]
    if method:
        text_lines += ["Method:", *[_plain_citations(line, refs) for line in method], ""]
    if rows:
        text_lines += ["Daily closes:"] + [f"  {r['date']:<11} {r['close']:>12} {r['change']:>10}" for r in rows]
        if frame is not None:
            text_lines.append(f"  {sparkline([float(c) for c in frame['close']])}")
    if visible_news:
        text_lines += ["", "Headlines:"] + [f"  [{n['ref'].number if n.get('ref') else '-'}] {n['title']} ({n['publisher']})" for n in visible_news]
    text_lines += ["", "Sources:"]
    text_lines += [f"  [{ref.number}] {ref.publisher}. {ref.title}{', ' + ref.date if ref.date else ''}.{' ' + ref.url if ref.url else ''}" for ref in sources]
    record = {**facts.record(), 'split_checks': (state.get('checks') or {}).get('splits') or [],
              'split_move': (state.get('checks') or {}).get('split_move'),
              'selection': state.get('selection') or {}, 'raw_top': state.get('raw_top') or {},
              'primary_announcements': announcements.matched(state),
              'screener_market_cap': (state.get('selected') or {}).get('market_cap'),
              'market_cap_status': 'unverified', 'renderer': renderer}
    return Report(subject=subject, html=page, text="\n".join(text_lines), images=images, notes=notes, facts=record,
                  renderer=renderer)

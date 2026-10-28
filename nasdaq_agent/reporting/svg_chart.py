"""Interactive inline SVG chart for the email.

Hovering a trading day shows a crosshair, highlighted points and a data card
(date, close, day change, change since the window start, benchmark day changes).
Pure CSS :hover - no JavaScript - plus a native <title> tooltip per day as backup.
Clients that render inline SVG (Apple Mail, iOS Mail, browsers) show it; clients
that strip SVG (Gmail, Outlook) get the static PNG instead: the template shows the
PNG unless `.chart-svg:has(svg)` matches, i.e. unless the SVG actually survived.
"""

from __future__ import annotations

import html
import math

import pandas as pd

from nasdaq_agent.reporting.chart import _series
from nasdaq_agent.reporting.placeholders import pct, pct_units

W, H = 592, 268
LEFT, RIGHT, TOP, BOTTOM = 44, 70, 34, 30
GRID, MUTED, INK = "#e2e8f0", "#64748b", "#0f172a"
UP, DOWN = "#15803d", "#b91c1c"
TIP_W, TIP_H = 168, 92

STYLE = """
.hz .hit{fill:#000;fill-opacity:0;pointer-events:all;cursor:crosshair}
.hz .xh,.hz .hl,.hz .tip{opacity:0}
.hz .tip{pointer-events:none}
.hz:hover .xh,.hz:hover .hl,.hz:hover .tip{opacity:1}
.hz:hover .hit{fill-opacity:.025}
"""


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def _ticks(lo: float, hi: float, target: int = 4) -> list[float]:
    raw = (hi - lo) / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]


def _pct(v: float | None) -> str:
    return "start" if v is None else pct(v)


def _money(v: float) -> str:
    return f"${v:,.4f}" if v < 1 else f"${v:,.2f}"


def _tooltip(i: int, x: float, series) -> list[str]:
    """Data card for day i, flipped to the left of the crosshair near the right edge."""
    left = x + 10 if x + 10 + TIP_W <= W - 4 else x - 10 - TIP_W
    top = TOP + 2
    stock_name, stock, _, _ = series[0]
    close = float(stock["close"].iloc[i])
    day = None if i == 0 else close / float(stock["close"].iloc[i - 1]) - 1
    since = close / float(stock["close"].iloc[0]) - 1
    date = pd.Timestamp(stock["date"].iloc[i]).strftime("%a %b %d").replace(" 0", " ")

    out = [f'<g class="tip" transform="translate({_fmt(left)},{top})">',
           f'<rect width="{TIP_W}" height="{TIP_H}" rx="8" fill="#0f172a"/>',
           f'<text x="10" y="17" font-size="11" font-weight="700" fill="#f8fafc">{date}</text>',
           f'<text x="10" y="34" font-size="11" fill="#cbd5e1">{html.escape(stock_name)} <tspan font-weight="700" fill="#f8fafc">{_money(close)}</tspan></text>',
           f'<text x="{TIP_W - 10}" y="34" font-size="11" text-anchor="end" font-weight="700" '
           f'fill="{"#4ade80" if (day or 0) >= 0 else "#f87171"}">{_pct(day)}</text>',
           f'<text x="10" y="50" font-size="10" fill="#94a3b8">since window start</text>',
           f'<text x="{TIP_W - 10}" y="50" font-size="10" text-anchor="end" fill="#e2e8f0">{_pct(since)}</text>']
    y = 68
    for name, data, color, _ in series[1:]:
        c = float(data["close"].iloc[i])
        d = None if i == 0 else c / float(data["close"].iloc[i - 1]) - 1
        out += [f'<rect x="10" y="{y - 7}" width="8" height="3" rx="1.5" fill="{color}"/>',
                f'<text x="22" y="{y - 3}" font-size="10" fill="#cbd5e1">{html.escape(name)}</text>',
                f'<text x="{TIP_W - 10}" y="{y - 3}" font-size="10" text-anchor="end" fill="#e2e8f0">{_pct(d)}</text>']
        y += 15
    out.append("</g>")
    return out


def svg_chart(datasets: dict, selected_id: str, days: int, *, frames: dict | None = None) -> str | None:
    if selected_id not in datasets:
        return None
    series = _series(datasets, selected_id, days, frames=frames)
    n = len(series[0][1])
    lo = min(s[1]["idx"].min() for s in series)
    hi = max(s[1]["idx"].max() for s in series)
    pad = (hi - lo) * 0.1 or 5
    lo, hi = lo - pad, hi + pad
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM

    def x(i: int) -> float:
        return LEFT + plot_w * i / (n - 1)

    def y(v: float) -> float:
        return TOP + plot_h * (1 - (v - lo) / (hi - lo))

    stock_name, stock, stock_color, _ = series[0]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="{html.escape(stock_name)} versus S&amp;P 500 and NASDAQ Composite, indexed to 100. Hover a day for details." '
             f'style="display:block;max-width:{W}px;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">',
             f"<style>{STYLE}</style>",
             '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{stock_color}" stop-opacity=".2"/><stop offset="1" stop-color="{stock_color}" stop-opacity="0"/>'
             "</linearGradient></defs>"]

    # grid + y labels on round steps (e.g. 100, 150, 200 ...)
    for v in _ticks(lo, hi):
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{_fmt(y(v))}" y2="{_fmt(y(v))}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{LEFT - 8}" y="{_fmt(y(v) + 4)}" text-anchor="end" font-size="10" fill="{MUTED}">{v:.0f}</text>')
    if lo < 100 < hi:
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{_fmt(y(100))}" y2="{_fmt(y(100))}" stroke="{MUTED}" '
                     'stroke-width="1" stroke-dasharray="3 4" opacity=".6"/>')
    for i, d in enumerate(stock["date"]):
        parts.append(f'<text x="{_fmt(x(i))}" y="{H - 10}" text-anchor="middle" font-size="10" fill="{MUTED}">'
                     f'{pd.Timestamp(d).strftime("%b %d")}</text>')

    # lines: benchmarks underneath, stock on top
    for j, (name, data, color, width) in reversed(list(enumerate(series))):
        pts = " ".join(f"{_fmt(x(i))},{_fmt(y(v))}" for i, v in enumerate(data["idx"]))
        if j == 0:
            area = f"{_fmt(x(0))},{_fmt(y(lo))} {pts} {_fmt(x(n - 1))},{_fmt(y(lo))}"
            parts.append(f'<polygon points="{area}" fill="url(#g)"/>')
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width + 0.4}" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
    for i, v in enumerate(stock["idx"]):
        parts.append(f'<circle cx="{_fmt(x(i))}" cy="{_fmt(y(v))}" r="3.5" fill="#fff" stroke="{stock_color}" stroke-width="2"/>')
    end = stock["idx"].iloc[-1]
    parts.append(f'<text x="{_fmt(x(n - 1) + 8)}" y="{_fmt(y(end) + 4)}" font-size="13" font-weight="700" '
                 f'fill="{stock_color}">{pct_units(end - 100)}</text>')

    # legend
    lx = LEFT
    for name, data, color, _ in series:
        label = f"{name} {pct_units(data['idx'].iloc[-1] - 100)}"
        parts.append(f'<rect x="{lx}" y="10" width="14" height="4" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="16" font-size="11" fill="{INK}">{html.escape(label)}</text>')
        lx += 30 + 7 * len(label)

    # hover zones last, so they sit above the lines and receive the pointer
    half = plot_w / (n - 1) / 2
    for i in range(n):
        cx = x(i)
        x0, x1 = max(LEFT - 6, cx - half), min(W - RIGHT + 6, cx + half)
        summary = []
        for name, data, _, _ in series:
            c = float(data["close"].iloc[i])
            d = None if i == 0 else c / float(data["close"].iloc[i - 1]) - 1
            summary.append(f"{name} {_money(c) if name == stock_name else f'{c:,.2f}'} ({_pct(d)})")
        date = pd.Timestamp(stock["date"].iloc[i]).strftime("%a %b %d")
        parts.append('<g class="hz">')
        parts.append(f"<title>{html.escape(date + ' · ' + ' · '.join(summary))}</title>")
        parts.append(f'<rect class="hit" x="{_fmt(x0)}" y="{TOP}" width="{_fmt(x1 - x0)}" height="{plot_h}"/>')
        parts.append(f'<line class="xh" x1="{_fmt(cx)}" x2="{_fmt(cx)}" y1="{TOP}" y2="{TOP + plot_h}" stroke="{MUTED}" '
                     'stroke-width="1" stroke-dasharray="2 3" pointer-events="none"/>')
        for _, data, color, width in series:
            parts.append(f'<circle class="hl" cx="{_fmt(cx)}" cy="{_fmt(y(data["idx"].iloc[i]))}" r="{5.5 if width > 2 else 4}" '
                         f'fill="{color}" stroke="#fff" stroke-width="2" pointer-events="none"/>')
        parts += _tooltip(i, cx, series)
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)

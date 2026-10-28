"""Static PNG chart rendered by the reporting layer (not by the agent).

Email clients that strip SVG (Gmail, Outlook) show this; others get the animated
SVG from ``svg_chart``. Both share the same series, colours and axes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nasdaq_agent.reporting.placeholders import pct_units  # noqa: E402

log = logging.getLogger(__name__)

INK, MUTED, GRID = "#0f172a", "#64748b", "#e2e8f0"
STOCK, SPX, NDX = "#16a34a", "#6366f1", "#f59e0b"
LABELS = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ Comp."}


def _series(datasets: dict, selected_id: str, days: int, *, frames: dict | None = None) -> list[tuple[str, pd.DataFrame, str, float]]:
    out = []
    for ds, meta in datasets.items():
        is_bench = bool(meta.get("proxy_for") or meta["symbol"].startswith("^"))
        if ds != selected_id and not is_bench:
            continue
        if frames is not None and ds not in frames:
            continue
        frame = (frames[ds].copy() if frames is not None else
                 pd.read_csv(meta["path"], parse_dates=["date"]).tail(days + 1).reset_index(drop=True))
        frame['date'] = pd.to_datetime(frame['date'])
        frame["idx"] = frame["close"] / frame["close"].iloc[0] * 100
        key = meta.get("proxy_for") or meta["symbol"]
        if ds == selected_id:
            out.insert(0, (meta["symbol"], frame, STOCK, 3.0))
        else:
            out.append((LABELS.get(key, key), frame, SPX if "GSPC" in key else NDX, 1.6))
    return out


def _axes(ax, series):
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    dates = series[0][1]["date"]
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.strftime("%b %d") for d in dates])
    ymax = max(s[1]["idx"].max() for s in series)
    ymin = min(s[1]["idx"].min() for s in series)
    pad = (ymax - ymin) * 0.12 or 5
    ax.set_ylim(ymin - pad, ymax + pad * 1.6)
    ax.set_xlim(-0.2, len(dates) - 0.6)
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.6)
    ax.set_ylabel("Indexed (start = 100)", color=MUTED, fontsize=9)


def normalised_chart(datasets: dict, selected_id: str, out: Path, days: int = 5, *, frames: dict | None = None) -> Path | None:
    if selected_id not in datasets:
        return None
    series = _series(datasets, selected_id, days, frames=frames)
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=180)  # 2x for high-DPI screens
    _axes(ax, series)
    for j, (name, data, color, width) in enumerate(series):
        change = data["idx"].iloc[-1] - 100
        label = name if j == 0 else f"{name}  {pct_units(change)}"
        ax.plot(range(len(data)), data["idx"], color=color, linewidth=width, marker="o",
                markersize=5 if j == 0 else 3, markerfacecolor="white" if j == 0 else color, label=label)
        if j == 0:
            ax.annotate(pct_units(change), (len(data) - 1, data["idx"].iloc[-1]), xytext=(-6, 6),
                        textcoords="offset points", ha="right", color=color, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out

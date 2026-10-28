"""Unicode sparkline - a chart that survives any email client."""

from __future__ import annotations

BARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return BARS[3] * len(values)
    return "".join(BARS[int((v - lo) / (hi - lo) * (len(BARS) - 1))] for v in values)

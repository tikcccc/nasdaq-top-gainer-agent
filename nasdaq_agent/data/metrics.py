"""Reference implementation of the analysis metrics.

Not used to produce the agent's analysis (the agent writes that code itself).
It is used to (1) cross-check the agent's numbers and (2) build the fallback
email when the agent cannot finish.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute(stock: pd.DataFrame, benchmarks: dict[str, pd.DataFrame], days: int, symbol: str) -> dict:
    if len(stock) < days + 1:
        raise ValueError(f"need {days + 1} closes for {days} returns")
    closes = stock["close"].tail(days + 1).reset_index(drop=True)
    returns = closes.pct_change().iloc[1:]
    cumulative = float(closes.iloc[-1] / closes.iloc[0] - 1)
    slope = float(np.polyfit(np.arange(len(closes)), closes / closes.iloc[0], 1)[0])
    trend = "up" if cumulative > 0.02 and slope > 0 else "down" if cumulative < -0.02 and slope < 0 else "sideways"
    running_max = closes.cummax()
    drawdown = float((closes / running_max - 1).min())
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    bench = {}
    for key, frame in benchmarks.items():
        c = frame["close"].tail(days + 1)
        bench[key.lstrip("^").upper()] = float(c.iloc[-1] / c.iloc[0] - 1)
    return {
        "symbol": symbol,
        "days": days,
        "daily_returns": [float(r) for r in returns],
        "avg_daily_return": float(returns.mean()),
        "cumulative_return": cumulative,
        "trend": trend,
        "trend_slope": slope,
        "volatility_daily": vol,
        "volatility_annualized": vol * math.sqrt(252),
        "max_drawdown": drawdown,
        "benchmark": bench,
        "excess_return": cumulative - bench["GSPC"] if "GSPC" in bench else None,
        "notes": "computed by the reference implementation (fallback)",
    }


def compare(agent: dict, reference: dict, tol: float = 1e-6) -> list[str]:
    """Differences between the agent's result and the reference that indicate a bug."""
    problems = []
    def number(key, a, r):
        if r is None:
            if a is not None:
                problems.append(f"{key} must be null when its benchmark is unavailable")
        elif a is None or not math.isfinite(a) or not math.isclose(a, r, rel_tol=1e-5, abs_tol=tol):
            problems.append(f"{key}={a!r}; expected {r:.8g}")

    for key in ("symbol", "days", "trend"):
        if agent.get(key) != reference.get(key):
            problems.append(f"{key}={agent.get(key)!r}; expected {reference.get(key)!r}")
    for key in ("cumulative_return", "avg_daily_return", "trend_slope", "volatility_daily",
                "volatility_annualized", "max_drawdown", "excess_return"):
        number(key, agent.get(key), reference.get(key))
    a_returns, r_returns = agent.get("daily_returns", []), reference["daily_returns"]
    if len(a_returns) != len(r_returns):
        problems.append("daily_returns length does not match the window")
    for i, (a, r) in enumerate(zip(a_returns, r_returns)):
        number(f"daily_returns[{i}]", a, r)
    a_bench, r_bench = agent.get("benchmark", {}), reference["benchmark"]
    if set(a_bench) != set(r_bench):
        problems.append(f"benchmark keys must be {sorted(r_bench)}")
    for key, r in r_bench.items():
        number(f"benchmark.{key}", a_bench.get(key), r)
    return problems

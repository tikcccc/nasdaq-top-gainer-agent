"""LangGraph state.

The LLM passes data between tools through tool arguments (symbol, dataset ids,
headline ids). Tools also write validated facts here, so code - not the LLM -
is the source of truth for numbers, guard checks and the fallback email.
Everything is plain JSON so the SQLite checkpointer can persist it.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


def add_list(left: list | None, right: list | None) -> list:
    return [*(left or []), *(right or [])]


def add_counts(left: dict | None, right: dict | None) -> dict:
    out = dict(left or {})
    for key, value in (right or {}).items():
        out[key] = out.get(key, 0) + value
    return out


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    run_id: str
    plan: dict[str, Any]                                  # {"version", "steps", "history"}
    session: dict[str, Any]                               # trade_date, is_intraday, previous_date
    candidates: list[dict[str, Any]]                      # last screener result shown to the agent
    raw_top: dict[str, Any]                               # unfiltered #1 for transparency
    selected: dict[str, Any]                              # the stock being analysed
    selection: dict[str, Any]                              # ranking scope and active filters
    ranking: dict[str, Any]                               # is the selected stock a verified #1? (separate from price checks)
    datasets: dict[str, dict[str, Any]]
    checks: Annotated[dict[str, Any], merge_dict]         # price consistency / eligibility / corporate actions
    analysis: dict[str, Any]
    headlines: Annotated[dict[str, dict[str, Any]], merge_dict]
    sentiment: dict[str, Any]
    charts: list[str]
    sources: Annotated[dict[str, str], merge_dict]        # data type -> provider actually used
    warnings: Annotated[list[str], add_list]
    errors: Annotated[list[dict[str, Any]], add_list]
    attempts: Annotated[dict[str, int], add_counts]
    failures: Annotated[dict[str, int], add_counts]
    usage: Annotated[dict[str, int], add_counts]
    llm_calls: Annotated[int, lambda a, b: (a or 0) + (b or 0)]
    guard_rejections: Annotated[int, lambda a, b: (a or 0) + (b or 0)]
    email: dict[str, Any]
    stop_reason: str
    degraded: Annotated[list[str], add_list]

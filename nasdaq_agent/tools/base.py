"""One response envelope for every tool.

Every tool returns ``{"ok", "data" | "error", "warnings", "source", "elapsed_ms"}``.
``error.retryable`` and ``error.hint`` are what let the agent self-correct:
they say whether repeating the call can help and what to do instead.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from nasdaq_agent.context import get_ctx
from nasdaq_agent.infra import trace
from nasdaq_agent.infra.errors import ProviderError, ToolError
from nasdaq_agent.infra.usage import BudgetExceeded

log = logging.getLogger(__name__)
MAX_CONTENT_CHARS = 6000


@dataclass
class ToolResult:
    data: dict[str, Any]
    update: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: str | None = None


def _content(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, default=str, ensure_ascii=False)
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + ' ... [truncated: output exceeded the tool size limit]'
    return text


def _plain(value: Any) -> Any:
    """JSON round-trip so state only ever holds plain types (numpy/pandas scalars break checkpointing)."""

    def default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    return json.loads(json.dumps(value, default=default))


def agent_tool(fn: Callable[..., ToolResult]) -> Callable[..., Command]:
    """Wrap a tool body: timing, error mapping, envelope, state bookkeeping, trace."""
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Command:
        tool_call_id = kwargs["tool_call_id"]
        started = time.time()
        update: dict[str, Any] = {"attempts": {name: 1}}
        try:
            if get_ctx().seconds_left() <= 0:
                raise BudgetExceeded("deadline")
            result = fn(*args, **kwargs)
            payload = {"ok": True, "data": result.data, "warnings": result.warnings, "source": result.source}
            update.update(_plain(result.update))
            if result.warnings:
                update["warnings"] = [f"[{name}] {w}" for w in result.warnings]
        except BudgetExceeded as exc:
            payload = {"ok": False, "error": {"type": "budget_exceeded", "message": str(exc), "retryable": False}}
            update["stop_reason"] = str(exc)
            update["failures"] = {name: 1}
            trace.write(get_ctx().run_dir, "budget_exceeded", dimension=exc.dimension)
        except ToolError as exc:
            payload = {"ok": False, "error": exc.as_dict()}
            update.update(_plain(exc.update))
            update["failures"] = {name: 1}
            state = kwargs.get("state") or {}
            primary = (state.get("selection") or {}).get("primary_symbol")
            if name == "get_price_history" and exc.type == "insufficient_history" and kwargs.get("symbol", "").upper() == primary:
                update["stop_reason"] = "the top-ranked stock has insufficient history for the requested window"
            if exc.stop_run:
                update["stop_reason"] = exc.stop_run
        except ProviderError as exc:
            payload = {"ok": False, "error": {
                "type": "provider_unavailable", "message": str(exc), "retryable": exc.retryable,
                "hint": "All data sources failed. Retry once; if it fails again continue with what you have "
                        "and say in the email which data is missing.",
            }}
            update["failures"] = {name: 1}
        except Exception as exc:  # noqa: BLE001 - never crash the graph from a tool
            log.exception("tool %s crashed", name)
            payload = {"ok": False, "error": {
                "type": "internal_error", "message": f"{type(exc).__name__}: {exc}", "retryable": False,
                "hint": "This is a bug in the tool, not in your input. Do not retry with the same arguments.",
            }}
            update["failures"] = {name: 1}

        payload["elapsed_ms"] = int((time.time() - started) * 1000)
        if name != "send_email" and get_ctx().seconds_left() <= 0:
            update["stop_reason"] = str(BudgetExceeded("deadline"))
        if not payload["ok"]:
            update["errors"] = [{"tool": name, **payload["error"]}]
        args_for_trace = {k: v for k, v in kwargs.items() if k not in {"state", "tool_call_id"}}
        trace.write(get_ctx().run_dir, "tool_result", tool=name, args=args_for_trace, ok=payload["ok"],
                    data=payload.get("data"), error=payload.get("error"), warnings=payload.get("warnings"),
                    elapsed_ms=payload["elapsed_ms"])
        update["messages"] = [ToolMessage(content=_content(payload), tool_call_id=tool_call_id, name=name,
                                          status="success" if payload["ok"] else "error")]
        return Command(update=update)

    return wrapper

"""Run (or resume) the agent and stream human-readable events.

Shared by the CLI and the web API. Each run writes to ``outputs/run_<id>/``:
trace.jsonl, agent.log, data/, sandbox/, code_v*.py, chart.png, email.eml/.html, summary.json.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from typing import Any, Callable, Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from nasdaq_agent.agent.fallback import fallback_node
from nasdaq_agent.agent.graph import build_graph
from nasdaq_agent.agent.prompts import task_message
from nasdaq_agent.config import Settings
from nasdaq_agent.context import RunContext, start_run, get_ctx
from nasdaq_agent.infra import breaker, trace
from nasdaq_agent.infra.schemas import UNVERIFIED_FLAGS
from nasdaq_agent.llm.factory import describe
from nasdaq_agent.tools.emailer import DELIVERED

log = logging.getLogger(__name__)
Event = dict[str, Any]

EXIT_OK, EXIT_FAILED, EXIT_DEGRADED = 0, 1, 2


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _events(node: str, update: Any) -> Iterator[Event]:
    """Translate one LangGraph node update into UI events."""
    updates = update if isinstance(update, list) else [update]
    for upd in updates:
        if not isinstance(upd, dict):
            continue
        for msg in upd.get("messages") or []:
            if isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content.strip():
                    yield {"type": "thought", "node": node, "text": msg.content.strip()}
                for call in msg.tool_calls:
                    yield {"type": "tool_call", "node": node, "tool": call["name"], "args": call["args"], "id": call["id"]}
            elif isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                except (TypeError, ValueError):
                    payload = {"ok": msg.status != "error", "raw": str(msg.content)[:500]}
                yield {"type": "tool_result", "node": node, "tool": msg.name, "id": msg.tool_call_id,
                       "ok": payload.get("ok"), "data": payload.get("data"), "error": payload.get("error"),
                       "warnings": payload.get("warnings") or [], "elapsed_ms": payload.get("elapsed_ms")}
        if upd.get("plan"):
            yield {"type": "plan", "node": node, "plan": upd["plan"]}
        if node == "guard" and "checks" in upd:
            yield {"type": "guard", "node": node, **upd["checks"].get("guard", {})}
        if upd.get("stop_reason") and node != "fallback":
            yield {"type": "stop", "node": node, "reason": upd["stop_reason"]}
        if node == "fallback" and upd.get("email"):
            yield {"type": "fallback", "node": node, "reason": upd.get("stop_reason"), "email": upd["email"]}


def _summary(ctx: RunContext, values: dict, started: float) -> dict:
    email = values.get("email") or {}
    delivered = email.get("status") in DELIVERED
    degraded = values.get("degraded") or []
    status = "failed" if not delivered else ("degraded" if degraded else "ok")
    selected = values.get("selected") or {}
    analysis = values.get("analysis") or {}
    primary = (values.get("selection") or {}).get("primary_symbol", selected.get("symbol"))
    requirements_met = bool(analysis) and analysis.get("days") == ctx.settings.analysis.days and selected.get("symbol") == primary
    requirements_met &= not bool(UNVERIFIED_FLAGS & set(selected.get("flags") or []))
    return {
        "run_id": ctx.run_id,
        "status": status,
        "exit_code": {"ok": EXIT_OK, "degraded": EXIT_DEGRADED}.get(status, EXIT_FAILED),
        "elapsed_s": round(time.time() - started, 1),
        "total_elapsed_s": round(time.time() - ctx.started_at, 1),
        "models": describe(ctx.settings),
        "sandbox": ctx.settings.sandbox.backend,
        "trade_date": (values.get("session") or {}).get("trade_date"),
        "raw_top": {k: (values.get("raw_top") or {}).get(k) for k in ("symbol", "pct_change", "flags")},
        "selection": values.get("selection"),
        "ranking": values.get("ranking"),
        "selected": {k: selected.get(k) for k in ("symbol", "name", "pct_change", "price", "volume", "flags", "days")},
        "analysis": values.get("analysis"),
        "sentiment": values.get("sentiment"),
        "headlines": values.get("headlines"),
        "email": {k: v for k, v in email.items() if k != "previous"},
        "plan": values.get("plan"),
        "checks": values.get("checks"),
        "sources": values.get("sources"),
        "llm_calls": values.get("llm_calls"),
        "usage": values.get("usage") or {},
        "model_usage": ctx.ledger.summary(ctx.run_id),
        "budget": ctx.settings.budget.model_dump(exclude={"prices"}),
        "tool_attempts": values.get("attempts"),
        "tool_failures": values.get("failures"),
        "guard_rejections": values.get("guard_rejections"),
        "warnings": values.get("warnings"),
        "errors": values.get("errors"),
        "degraded": degraded,
        "stop_reason": values.get("stop_reason"),
        "charts": [p.split("\\")[-1].split("/")[-1] for p in values.get("charts") or []],
        "analysis_completed": bool(values.get("analysis")),
        "agent_completed": delivered and requirements_met and not degraded and not email.get("fallback"),
        "task_completed": requirements_met and not degraded and email.get("status") == "sent" and not email.get("fallback"),
    }


def _run_current(
    settings: Settings,
    *,
    checkpoint_conn,
    resume_id: str | None = None,
    run_id: str | None = None,
    force_send: bool = False,
    on_event: Callable[[Event], None] | None = None,
) -> dict:
    """Start a new run (optionally with a caller-chosen ``run_id``) or resume ``resume_id``."""
    ctx = get_ctx()
    ctx.force_send = force_send
    breaker.reset(settings.run.breaker_failures)
    callback = on_event or (lambda e: None)
    def emit(event):
        # Durable UI events support replay after API restarts and SSE reconnection.
        with (ctx.run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
        callback(event)
    started = time.time()
    emit({"type": "run_started", "run_id": ctx.run_id, "resumed": bool(resume_id), "models": describe(settings),
          "dry_run": settings.email.dry_run, "recipients": settings.email.to})
    trace.write(ctx.run_dir, "run_started", resumed=bool(resume_id), models=describe(settings))

    conn = checkpoint_conn
    graph = build_graph(SqliteSaver(conn))
    config = {"configurable": {"thread_id": ctx.run_id}, "recursion_limit": settings.run.recursion_limit}
    inputs = None if resume_id else {"messages": [HumanMessage(task_message(settings))], "run_id": ctx.run_id}

    try:
        for chunk in graph.stream(inputs, config, stream_mode="updates"):
            for node, update in chunk.items():
                for event in _events(node, update):
                    emit(event)
    except GraphRecursionError:
        emit({"type": "stop", "reason": "graph recursion limit reached"})
        values = graph.get_state(config).values
        update = fallback_node({**values, "stop_reason": "graph recursion limit reached"})
        graph.update_state(config, update, as_node="fallback")
        for event in _events("fallback", update):
            emit(event)
    except Exception as exc:  # noqa: BLE001 - last line of defence: still try the fallback email
        log.exception("run crashed")
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        values = graph.get_state(config).values
        update = fallback_node({**values, "stop_reason": f"unexpected error: {type(exc).__name__}"})
        graph.update_state(config, update, as_node="fallback")
        for event in _events("fallback", update):
            emit(event)

    values = graph.get_state(config).values
    summary = _summary(ctx, values, started)
    ctx.ledger.finish_run(summary)
    pending = ctx.run_dir / "summary.pending.json"
    pending.write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    pending.replace(ctx.run_dir / "summary.json")
    trace.write(ctx.run_dir, "run_finished", status=summary["status"], elapsed_s=summary["elapsed_s"])
    emit({"type": "run_finished", "summary": summary})
    return summary


def run(settings: Settings, *, resume_id: str | None = None, run_id: str | None = None,
        force_send: bool = False, on_event: Callable[[Event], None] | None = None) -> dict:
    ctx = start_run(settings, resume_id or run_id)
    with ctx.ledger.session(), trace.run_logging(ctx.run_dir), closing(sqlite3.connect(settings.output_root / "checkpoints.sqlite", check_same_thread=False)) as conn:
        return _run_current(settings, checkpoint_conn=conn, resume_id=resume_id, run_id=run_id,
                            force_send=force_send, on_event=on_event)

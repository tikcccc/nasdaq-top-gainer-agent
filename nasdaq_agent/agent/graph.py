"""The LangGraph agent.

    START -> agent --(tool calls)--> tools -> agent ...   (tools -> END once the email is delivered)
                   --(send_email)--> guard --pass--> tools
                                           --reject--> agent
                   --(budget/deadline/LLM down/stopped early)--> fallback -> END
                   --(done)--> END

The LLM decides the order of tool calls; code only routes, guards the outward
action and guarantees a report.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from nasdaq_agent.agent import guard
from nasdaq_agent.agent.fallback import fallback_node
from nasdaq_agent.agent.prompts import system_prompt
from nasdaq_agent.agent.state import AgentState
from nasdaq_agent.context import get_ctx
from nasdaq_agent.infra import trace
from nasdaq_agent.infra.usage import BudgetExceeded
from nasdaq_agent.llm.factory import build_chat
from nasdaq_agent.tools import ALL_TOOLS
from nasdaq_agent.tools.emailer import DELIVERED

log = logging.getLogger(__name__)


def _llm(run_id: str):
    ctx = get_ctx()
    if ctx.model is None:
        ctx.model = build_chat(ctx.settings, ALL_TOOLS)
    return ctx.model


def _prompt(run_id: str) -> str:
    """Built once per run: an identical prefix on every call lets the providers reuse their prompt cache."""
    ctx = get_ctx()
    if ctx.prompt is None:
        ctx.prompt = system_prompt(ctx.settings, datetime.fromtimestamp(ctx.started_at, tz=timezone.utc))
    return ctx.prompt


def agent_node(state: AgentState) -> dict:
    ctx = get_ctx()
    settings = ctx.settings
    if ctx.seconds_left() <= 0:
        return {"stop_reason": str(BudgetExceeded("deadline"))}
    if (state.get("llm_calls") or 0) >= settings.run.max_llm_calls:
        return {"stop_reason": str(BudgetExceeded("agent_turns"))}
    messages = [SystemMessage(_prompt(ctx.run_id)), *state["messages"]]
    try:
        reply = _llm(ctx.run_id).invoke(messages)
    except BudgetExceeded as exc:
        trace.write(ctx.run_dir, "budget_exceeded", dimension=exc.dimension)
        return {"stop_reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 - every provider in the fallback chain failed
        log.exception("all LLM providers failed")
        return {"stop_reason": f"LLM unavailable ({type(exc).__name__})", "llm_calls": 1}
    trace.write(ctx.run_dir, "llm", content=reply.content if isinstance(reply.content, str) else str(reply.content),
                tool_calls=[{"name": c["name"], "args": c["args"]} for c in reply.tool_calls],
                provider=(reply.response_metadata or {}).get("model_name"))
    update = {"messages": [reply], "llm_calls": 1,
              "usage": {k: int(v) for k, v in (reply.usage_metadata or {}).items() if isinstance(v, int)}}
    if ctx.seconds_left() <= 0:
        update["stop_reason"] = str(BudgetExceeded("deadline"))
    return update


def route_after_agent(state: AgentState) -> str:
    if state.get("stop_reason"):
        return "fallback"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "guard" if any(c["name"] == "send_email" for c in last.tool_calls) else "tools"
    if (state.get("email") or {}).get("status") in DELIVERED:
        return END
    return "fallback"


def route_after_tools(state: AgentState) -> str:
    """Stop as soon as the report is delivered: no extra LLM round trip for a closing remark."""
    if state.get("stop_reason"):
        return "fallback"
    return END if (state.get("email") or {}).get("status") in DELIVERED else "agent"


def guard_node(state: AgentState) -> Command:
    ctx = get_ctx()
    last = state["messages"][-1]
    email_call = next(c for c in last.tool_calls if c["name"] == "send_email")
    if len(last.tool_calls) != 1:
        problems, details = ["send_email must be the only tool call in its turn; send exactly one verified report"], {}
    else:
        try:
            problems, details = guard.check(state, email_call["args"], ctx.settings)
        except BudgetExceeded as exc:
            trace.write(ctx.run_dir, "budget_exceeded", dimension=exc.dimension)
            return Command(goto="fallback", update={"stop_reason": str(exc)})
        if ctx.seconds_left() <= 0:
            return Command(goto="fallback", update={"stop_reason": str(BudgetExceeded("deadline"))})
    trace.write(ctx.run_dir, "guard", passed=not problems, problems=problems, **details)
    if not problems:
        return Command(goto="tools", update={"checks": {"guard": {"passed": True, **details}}})

    rejections = (state.get("guard_rejections") or 0) + 1
    messages = [ToolMessage(guard.rejection_payload(problems), tool_call_id=email_call["id"], name="send_email", status="error")]
    messages += [
        ToolMessage('{"ok": false, "error": {"type": "not_executed", "message": "skipped because send_email in the same '
                    'turn was rejected by the pre-send guard", "retryable": true}}',
                    tool_call_id=c["id"], name=c["name"], status="error")
        for c in last.tool_calls if c["id"] != email_call["id"]
    ]
    update = {
        "messages": messages,
        "guard_rejections": 1,
        "errors": [{"tool": "guard", "type": "guard_rejected", "message": " | ".join(problems)}],
        "checks": {"guard": {"passed": False, "problems": problems, **details}},
    }
    if any(p.startswith("semantic_checker_unavailable") for p in problems):
        update["stop_reason"] = "semantic checker unavailable; sending a deterministic report instead"
        return Command(goto="fallback", update=update)
    if rejections > ctx.settings.run.max_guard_rejections:
        update["stop_reason"] = f"pre-send guard rejected the report selection {rejections} times"
        return Command(goto="fallback", update=update)
    return Command(goto="agent", update=update)


def build_graph(checkpointer=None):
    tools = ToolNode(ALL_TOOLS, handle_tool_errors=True)
    def execute_tools(state: AgentState, config: RunnableConfig):
        calls = state["messages"][-1].tool_calls
        names = [c["name"] for c in calls]
        conflict = len(names) != len(set(names))
        conflict |= "get_top_nasdaq_gainers" in names and any(n not in {"record_plan", "get_top_nasdaq_gainers"} for n in names)
        conflict |= "get_price_history" in names and any(n in {"execute_python", "analyze_sentiment"} for n in names)
        conflict |= "get_news" in names and "analyze_sentiment" in names
        if conflict:
            return {"messages": [ToolMessage('{"ok":false,"error":{"type":"conflicting_batch","retryable":true,"message":"dependent or duplicate tools cannot execute in the same turn"}}',
                                              tool_call_id=c["id"], name=c["name"], status="error") for c in calls]}
        return tools.invoke(state, config)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", execute_tools)
    builder.add_node("guard", guard_node, destinations=("tools", "agent", "fallback"))
    builder.add_node("fallback", fallback_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "guard", "fallback", END])
    builder.add_conditional_edges("tools", route_after_tools, ["agent", "fallback", END])
    builder.add_edge("fallback", END)
    return builder.compile(checkpointer=checkpointer, name="nasdaq-top-gainer-agent")

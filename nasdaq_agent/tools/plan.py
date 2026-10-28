"""record_plan: makes the agent's planning explicit and auditable."""

from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from nasdaq_agent.tools.base import ToolResult, agent_tool


@tool("record_plan")
@agent_tool
def record_plan(
    steps: Annotated[list[str], Field(min_length=1, max_length=12, description="Ordered steps you intend to take, each a short sentence naming the tool you will use.")],
    revise: Annotated[bool, Field(description="True when changing an existing plan because of something you learned.")] = False,
    reason: Annotated[str | None, Field(description="Why the plan changed (required when revise=true).")] = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> ToolResult:
    """Record or revise your plan. Call this FIRST, before any other tool, and again with
    revise=true whenever results force a change of approach (e.g. the top gainer is a
    warrant, data is missing, analysis code keeps failing). The plan is shown in the run log;
    it is not enforced, so keep it honest and short."""
    previous = (state or {}).get("plan") or {}
    if revise and not reason:
        reason = "unspecified"
    version = previous.get("version", 0) + 1
    history = list(previous.get("history", []))
    if previous:
        history.append({"version": previous.get("version"), "steps": previous.get("steps"), "replaced_because": reason})
    plan = {"version": version, "steps": steps, "reason": reason, "history": history}
    return ToolResult(data={"plan_version": version, "steps": steps}, update={"plan": plan})

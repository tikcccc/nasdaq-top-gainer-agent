"""Entry point for LangGraph Studio (`langgraph dev`): the same graph, with a run context."""

from nasdaq_agent.agent.graph import build_graph
from nasdaq_agent.config import get_settings
from nasdaq_agent.context import start_run
from nasdaq_agent.infra import breaker

_settings = get_settings()
start_run(_settings, "studio")
breaker.reset(_settings.run.breaker_failures)

# Studio supplies its own checkpointer and thread management.
graph = build_graph()

"""Typed errors that tools turn into actionable messages for the LLM."""

from __future__ import annotations


class ToolError(Exception):
    """An expected failure with guidance for the agent.

    ``retryable`` tells the LLM whether repeating the same call can succeed;
    ``hint`` tells it what to do instead. ``stop_run`` ends the agent loop with that
    reason and hands over to the deterministic fallback, for failures no model action can fix.
    ``update`` is state the failure still establishes (e.g. a candidate ruled out).
    """

    def __init__(self, type_: str, message: str, *, retryable: bool = False, hint: str | None = None,
                 stop_run: str | None = None, update: dict | None = None):
        super().__init__(message)
        self.type = type_
        self.message = message
        self.retryable = retryable
        self.hint = hint
        self.stop_run = stop_run
        self.update = update or {}

    def as_dict(self) -> dict:
        return {"type": self.type, "message": self.message, "retryable": self.retryable, "hint": self.hint}


class ProviderError(Exception):
    """A data provider failed (network, bad payload, stale data...)."""

    def __init__(self, provider: str, message: str, *, retryable: bool = True):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.retryable = retryable


class StaleDataError(ProviderError):
    def __init__(self, provider: str, message: str):
        super().__init__(provider, message, retryable=False)

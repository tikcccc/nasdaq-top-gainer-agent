"""Jev (TypeSafe System One) client.

Jev does not generate text; it answers typed questions (choice / score / noul)
about a supplied state with calibrated probabilities. We use it for the narrow
judgements in this agent: news sentiment and narrative-vs-data consistency.

The SDK is a recommended extra (`uv sync --extra jev`, included in `--extra ui`).
Without it, or without TYPESAFE_API_KEY, sentiment falls back to the chat model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

try:
    from typesafe_sdk import Choice, Noul, RetryPolicy, Score, SystemOneResponse, TypeSafeClient
except ImportError:  # optional dependency
    Choice = Noul = RetryPolicy = Score = SystemOneResponse = TypeSafeClient = None

from nasdaq_agent.config import Settings
from nasdaq_agent.infra.usage import BudgetExceeded, estimate_tokens
from nasdaq_agent.llm import tracked

__all__ = ["Choice", "Noul", "Score", "available", "ask", "status"]
SDK_INSTALLED = TypeSafeClient is not None


def available(settings: Settings) -> bool:
    return bool(SDK_INSTALLED and settings.jev.enabled and settings.jev.api_key)


def status(settings: Settings, last_request: dict | None = None) -> dict:
    """Configuration state, refined by the last recorded Jev request when there is one:
    a configured key the service rejected is reported as ``auth_failed``."""
    if not settings.jev.enabled:
        state = "disabled"
    elif not SDK_INSTALLED:
        state = "sdk_missing"
    elif not settings.jev.api_key:
        state = "key_missing"
    elif last_request and last_request.get("outcome") == "error" and (
            last_request.get("http_status") in (401, 403) or "Authentication" in (last_request.get("error_type") or "")):
        state = "auth_failed"
    else:
        state = "configured"
    return {"state": state, "enabled": settings.jev.enabled, "sdk_installed": SDK_INSTALLED,
            "api_key_configured": bool(settings.jev.api_key), "model": settings.jev.model,
            "last_request": last_request}


@lru_cache(maxsize=1)
def _client(api_key: str, base_url: str, model: str) -> "TypeSafeClient":
    return TypeSafeClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        retry=RetryPolicy(max_retries=0),
        timeout=20.0,
    )


def ask(settings: Settings, state: Any, questions: Mapping[str, Any], *, stage: str = "semantic_guard") -> "SystemOneResponse":
    if not available(settings):
        raise RuntimeError(f"Jev unavailable: {status(settings)['state']}")
    client = _client(settings.jev.api_key.get_secret_value(), settings.jev.base_url, settings.jev.model)
    input_estimate = estimate_tokens({"state": state, "questions": questions})
    for attempt in range(1, 5):
        try:
            return tracked.request(
                settings, provider="jev", model=settings.jev.model, stage=stage, attempt=attempt,
                input_estimate=input_estimate, output_estimate=max(256, len(questions) * 128), timeout_s=20,
                invoke=lambda timeout: client.system_one(state=state, questions=questions, timeout=timeout))
        except BudgetExceeded:
            raise
        except Exception as exc:
            if attempt == 4 or not tracked.retryable(exc):
                raise
            tracked.wait_before_retry(exc, attempt)

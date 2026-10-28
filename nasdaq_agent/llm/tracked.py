"""One accounted request per SDK call; retries stay outside SDKs."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import APIConnectionError, APIStatusError, APITimeoutError
try:
    from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError
except ImportError:  # optional Jev SDK: no Jev requests can raise these
    class TypeSafeAPIConnectionError(Exception):
        pass

    class TypeSafeAPIError(Exception):
        status = 0

from nasdaq_agent.context import get_ctx
from nasdaq_agent.infra import trace
from nasdaq_agent.infra.usage import BudgetExceeded


def retryable(exc):
    if isinstance(exc, BudgetExceeded):
        return False
    return (isinstance(exc, (APIConnectionError, APITimeoutError))
            or isinstance(exc, APIStatusError) and (exc.status_code in {408, 409, 429} or exc.status_code >= 500)
            or isinstance(exc, TypeSafeAPIConnectionError)
            or isinstance(exc, TypeSafeAPIError) and (exc.status in {408, 429} or exc.status >= 500))


def wait_before_retry(exc, attempt):
    """Respect Retry-After without waiting through the run deadline."""
    headers = getattr(exc, "headers", None) or getattr(getattr(exc, "response", None), "headers", {})
    headers = {str(k).lower(): v for k, v in headers.items()}
    delay = None
    try:
        if headers.get("retry-after-ms"):
            delay = float(headers["retry-after-ms"]) / 1000
        elif headers.get("retry-after"):
            try:
                delay = float(headers["retry-after"])
            except ValueError:
                delay = (parsedate_to_datetime(headers["retry-after"]) - datetime.now(timezone.utc)).total_seconds()
    except (ValueError, TypeError, OverflowError):
        pass
    if delay is None:
        delay = min(0.5 * 2 ** (attempt - 1), 5) * random.uniform(0.75, 1)
    delay = max(0, delay)
    if get_ctx().seconds_left() <= delay:
        raise BudgetExceeded("deadline")
    time.sleep(delay)


def request(settings, *, provider, model, stage, attempt, fallback=False,
            input_estimate, output_estimate, timeout_s, invoke):
    ctx = get_ctx()
    call_id = ctx.ledger.reserve(ctx.run_id, settings, provider=provider, model=model, stage=stage,
                                 attempt=attempt, fallback=fallback,
                                 input_estimate=input_estimate, output_estimate=output_estimate)
    started = time.monotonic()
    response, error, interrupted = None, None, False
    try:
        remaining = ctx.seconds_left()
        if remaining <= 0:
            raise BudgetExceeded("deadline")
        response = invoke(min(timeout_s, remaining))
        if isinstance(response, dict) and response.get("parsing_error"):
            raise response["parsing_error"]
        return response
    except Exception as exc:
        error = exc
        raise
    except BaseException:
        interrupted = True
        raise
    finally:
        if not interrupted:
            elapsed = round((time.monotonic() - started) * 1000, 3)
            ctx.ledger.finish(call_id, latency_ms=elapsed, response=response, error=error)
            trace.write(ctx.run_dir, "model_request", call_id=call_id, stage=stage, provider=provider,
                        model=model, attempt=attempt, fallback=fallback, latency_ms=elapsed,
                        outcome="error" if error else "success", error_type=type(error).__name__ if error else None)

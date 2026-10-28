"""Provider chains with a per-run circuit breaker.

Each data type has an ordered list of providers. ``first_success`` tries them in
order and records which one answered. A provider that fails ``breaker_failures``
times in a row is skipped for the rest of the run, so one dead API cannot keep
eating the run's time budget.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextvars import ContextVar
import threading
from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar

from nasdaq_agent.infra.errors import ProviderError, ToolError

log = logging.getLogger(__name__)
T = TypeVar("T")

_state: ContextVar[tuple[dict[str, int], int, threading.Lock] | None] = ContextVar("provider_breaker", default=None)


def _bucket():
    state = _state.get()
    if state is None:
        reset()
        state = _state.get()
    return state


def reset(threshold: int = 3) -> None:
    _state.set((defaultdict(int), threshold, threading.Lock()))


def is_open(provider: str) -> bool:
    failures, threshold, lock = _bucket()
    with lock:
        return failures[provider] >= threshold


def health() -> dict[str, int]:
    failures, _, lock = _bucket()
    with lock:
        return dict(failures)


@dataclass
class ChainResult(Generic[T]):
    value: T
    source: str
    warnings: list[str] = field(default_factory=list)


def first_success(attempts: Sequence[tuple[str, Callable[[], T]]]) -> ChainResult[T]:
    """Run provider callables in order; return the first result that doesn't raise."""
    warnings: list[str] = []
    errors: list[str] = []
    causes: list[Exception] = []
    for name, fn in attempts:
        if is_open(name):
            warnings.append(f"{name} skipped (circuit open after repeated failures)")
            continue
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - every provider failure is handled the same way
            causes.append(exc)
            failures, _, lock = _bucket()
            with lock:
                failures[name] += 1
            message = str(exc) if isinstance(exc, ProviderError) else f"{name}: {type(exc).__name__}: {exc}"
            log.warning("provider failed: %s", message)
            errors.append(message)
            warnings.append(f"{name} failed: {message[:160]}")
            continue
        failures, _, lock = _bucket()
        with lock:
            failures[name] = 0
        return ChainResult(value=value, source=name, warnings=warnings)
    if causes and all(isinstance(exc, ToolError) and not exc.retryable and exc.type == causes[0].type for exc in causes):
        raise causes[0]
    raise ProviderError("all", "; ".join(errors) or "no provider available", retryable=True)

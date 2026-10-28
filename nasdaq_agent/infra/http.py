"""Shared HTTP client: timeouts, per-host rate limiting and retry with backoff.

Only transient failures are retried (timeouts, connection errors, 429, 5xx).
A ``Retry-After`` header on 429/503 is honoured before the next attempt.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; nasdaq-agent/0.1; take-home assignment; low-volume research use)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Minimum seconds between requests per host, to stay polite with public endpoints.
MIN_INTERVAL = {"api.nasdaq.com": 1.0, "news.google.com": 1.0}


class RetryableHTTPError(Exception):
    def __init__(self, status: int, retry_after: float | None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


class _RateLimiter:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        interval = MIN_INTERVAL.get(host, 0.0)
        if not interval:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last.get(host, 0.0)
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last[host] = time.monotonic()


_limiter = _RateLimiter()
_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=httpx.Timeout(20.0, connect=5.0),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/xml, */*"},
            follow_redirects=True,
        )
    return _client


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RetryableHTTPError, httpx.TimeoutException, httpx.TransportError))


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, RetryableHTTPError) and exc.retry_after:
        return min(exc.retry_after, 30.0)
    return wait_exponential_jitter(initial=1, max=15)(state)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning("HTTP retry %s after %s", state.attempt_number, exc)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=_wait,
    stop=stop_after_attempt(4),
    before_sleep=_log_retry,
    reraise=True,
)
def get(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
    _limiter.wait(urlparse(url).hostname or "")
    response = client().get(url, params=params, headers=headers)
    if response.status_code in RETRYABLE_STATUS:
        retry_after = response.headers.get("Retry-After")
        raise RetryableHTTPError(
            response.status_code,
            float(retry_after) if retry_after and retry_after.isdigit() else None,
        )
    response.raise_for_status()
    return response


def get_json(url: str, **kwargs: Any) -> Any:
    return get(url, **kwargs).json()

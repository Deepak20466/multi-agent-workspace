"""Retry resilience layer built on tenacity.

Wraps flaky I/O (LLM calls, vector store calls, external APIs) with
exponential backoff + jitter, circuit-breaker-style failure tracking,
and structured logging of each attempt.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from typing import Callable, TypeVar

from prometheus_client import Counter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
    before_sleep_log,
    RetryCallState,
)

logger = logging.getLogger("retry_handler")

API_RETRY_COUNTER = Counter(
    "api_retries_total",
    "Total number of external API call retries due to rate limits or connection errors",
    ["function"],
)

T = TypeVar("T")

MAX_ATTEMPTS = int(os.getenv("MAX_API_RETRIES", "5"))
BASE_DELAY = float(os.getenv("BACKOFF_FACTOR", "1"))
MAX_BACKOFF = float(os.getenv("MAX_BACKOFF_S", "10"))


class RetryableError(Exception):
    """Raised by callers to mark an error as safe to retry."""


class RateLimitError(Exception):
    """Raised when an LLM/embedding provider responds with a rate-limit error."""


class APIConnectionError(Exception):
    """Raised when a network-level failure occurs talking to an external API."""


class CircuitBreaker:
    """A simple per-key circuit breaker to stop hammering a dead dependency."""

    def __init__(self, failure_threshold: int = 5, reset_after_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_after_seconds = reset_after_seconds
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_open(self, key: str) -> bool:
        with self._lock:
            opened_at = self._opened_at.get(key)
            if opened_at is None:
                return False
            if time.monotonic() - opened_at > self.reset_after_seconds:
                self._failures[key] = 0
                del self._opened_at[key]
                return False
            return True

    def record_failure(self, key: str) -> None:
        with self._lock:
            count = self._failures.get(key, 0) + 1
            self._failures[key] = count
            if count >= self.failure_threshold:
                self._opened_at[key] = time.monotonic()

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures[key] = 0
            self._opened_at.pop(key, None)


_breaker = CircuitBreaker()


class CircuitOpenError(Exception):
    """Raised when a call is refused because its circuit breaker is open."""


def _log_attempt(retry_state: RetryCallState) -> None:
    logger.warning(
        "retry attempt=%s fn=%s outcome=%s",
        retry_state.attempt_number,
        retry_state.fn.__name__ if retry_state.fn else "?",
        retry_state.outcome,
    )


def with_retry(
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY,
    max_backoff: float = MAX_BACKOFF,
    exceptions: tuple[type[Exception], ...] = (RetryableError, ConnectionError, TimeoutError),
    breaker_key: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator applying exponential backoff + jitter and an optional circuit breaker."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        key = breaker_key or func.__name__

        tenacity_retry = retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=base_delay, max=max_backoff),
            retry=retry_if_exception_type(exceptions),
            before_sleep=_log_attempt,
        )

        @tenacity_retry
        def _call(*args, **kwargs):
            return func(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if _breaker.is_open(key):
                raise CircuitOpenError(f"circuit open for '{key}', refusing call")
            try:
                result = _call(*args, **kwargs)
            except exceptions:
                _breaker.record_failure(key)
                raise
            else:
                _breaker.record_success(key)
                return result

        return wrapper

    return decorator


def _rate_limit_before_sleep(retry_state: RetryCallState) -> None:
    fn_name = retry_state.fn.__name__ if retry_state.fn else "?"
    API_RETRY_COUNTER.labels(function=fn_name).inc()
    before_sleep_log(logger, logging.WARNING)(retry_state)


_RATE_LIMIT_RETRY_KWARGS = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    before_sleep=_rate_limit_before_sleep,
    reraise=True,
)


def api_rate_limit_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Retry a synchronous API call up to 5x with exponential backoff on
    RateLimitError/APIConnectionError, incrementing `api_retries_total`
    on every retry.
    """

    return retry(**_RATE_LIMIT_RETRY_KWARGS)(fn)


def async_api_rate_limit_retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Async counterpart of api_rate_limit_retry.

    tenacity's `retry` decorator natively detects coroutine functions and
    awaits them correctly, so the policy is identical — this wrapper
    exists so call sites can self-document whether the wrapped function
    is sync or async.
    """

    return retry(**_RATE_LIMIT_RETRY_KWARGS)(fn)

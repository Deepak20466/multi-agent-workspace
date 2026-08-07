"""Structured logging + Prometheus metrics for the whole workspace.

Every agent/tool call should be wrapped with `traced_call` so latency,
success/failure, and route are uniformly observable regardless of
which agent handled the request.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

from loguru import logger
from prometheus_client import Counter, Histogram

LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
    ],
)

logger.remove()
logger.add(sys.stderr, serialize=True, level="INFO")
logger.add(os.path.join(LOG_DIR, "telemetry.log"), serialize=True, level="INFO", rotation="10 MB")

T = TypeVar("T")

REQUEST_COUNTER = Counter(
    "agent_requests_total", "Total requests handled per agent/route", ["route", "status"]
)
REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds", "Latency of agent/route calls", ["route"]
)
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds", "Latency of retrieval calls", ["strategy"]
)


@contextmanager
def traced_call(route: str) -> Iterator[None]:
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(route=route).observe(elapsed)
        REQUEST_COUNTER.labels(route=route, status=status).inc()
        logger.bind(route=route, status=status, latency_ms=round(elapsed * 1000, 2)).info("traced_call")


def traced(route: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of traced_call for functions/agent entrypoints."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with traced_call(route):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_event(event: str, **kwargs) -> None:
    logger.bind(**kwargs).info(event)

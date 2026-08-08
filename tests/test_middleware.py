"""Tests for `src.middleware.RateLimiter` (Redis token-bucket, fails open
on any Redis problem) and `APIKeyMiddleware`.

Redis isn't running in this environment, so the Redis-available cases
mock `redis.asyncio.Redis`/its Lua script call for determinism -- see
the task report for what was and wasn't live-tested against a real
Redis instance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.middleware import RateLimiter


# --- Redis unavailable: fails open -------------------------------------------


async def test_rate_limiter_allows_when_no_redis_url(monkeypatch):
    # RateLimiter falls back to os.getenv("REDIS_URL") when redis_url=None
    # (matching the CLI's usage) -- other tests importing main.py can leak
    # a real REDIS_URL from .env via load_dotenv(), so isolate explicitly
    # rather than relying on ambient absence.
    monkeypatch.delenv("REDIS_URL", raising=False)

    limiter = RateLimiter(redis_url=None)
    assert limiter._redis is None
    assert await limiter.allow("user-1") is True


async def test_rate_limiter_allows_when_redis_construction_fails():
    """The reported degrade-gracefully contract: a Redis connect failure
    at construction must not raise -- the limiter just fails open.
    """

    with patch("redis.asyncio.from_url", side_effect=ConnectionError("connection refused")):
        limiter = RateLimiter(redis_url="redis://localhost:6379/0")

    assert limiter._redis is None
    assert await limiter.allow("user-1") is True


async def test_rate_limiter_allows_when_script_call_fails():
    """A Redis connection that was fine at startup but fails mid-call
    (network blip, Redis restart) must also fail open, not 500 every
    request until the process restarts.
    """

    mock_redis = MagicMock()
    mock_script = AsyncMock(side_effect=ConnectionError("connection reset"))
    mock_redis.register_script.return_value = mock_script

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        limiter = RateLimiter(redis_url="redis://localhost:6379/0")

    assert await limiter.allow("user-1") is True


# --- Redis available (mocked): genuine token-bucket behavior ----------------


async def test_rate_limiter_uses_redis_script_when_available():
    mock_redis = MagicMock()
    mock_script = AsyncMock(return_value=1)
    mock_redis.register_script.return_value = mock_script

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        limiter = RateLimiter(redis_url="redis://localhost:6379/0", capacity=60, window_seconds=60.0)

    assert limiter._redis is mock_redis

    allowed = await limiter.allow("user-1")

    assert allowed is True
    mock_script.assert_awaited_once()
    _, kwargs = mock_script.call_args
    assert kwargs["keys"] == ["ratelimit:user-1"]
    assert kwargs["args"][0] == 60  # capacity


async def test_rate_limiter_denies_when_script_reports_no_tokens():
    mock_redis = MagicMock()
    mock_script = AsyncMock(return_value=0)
    mock_redis.register_script.return_value = mock_script

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        limiter = RateLimiter(redis_url="redis://localhost:6379/0")

    assert await limiter.allow("user-1") is False


async def test_rate_limiter_keys_are_scoped_per_caller():
    mock_redis = MagicMock()
    mock_script = AsyncMock(return_value=1)
    mock_redis.register_script.return_value = mock_script

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        limiter = RateLimiter(redis_url="redis://localhost:6379/0")

    await limiter.allow("alice")
    await limiter.allow("bob")

    called_keys = [call.kwargs["keys"][0] for call in mock_script.await_args_list]
    assert called_keys == ["ratelimit:alice", "ratelimit:bob"]

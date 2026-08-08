"""Tests for `src.cache.ResponseCache`: Redis-backed with an in-process
LRU fallback so a Redis outage degrades the cache tier instead of taking
the app down.

Redis isn't running in this environment (no `redis-server`/Docker/WSL
available), so every test here mocks the `redis` client for determinism
-- see the task report for what was and wasn't live-tested against a
real Redis instance.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.cache import DEFAULT_TTL_SECONDS, ResponseCache, _LRUCache


# --- _LRUCache (pure in-memory logic, no mocking needed) --------------------


def test_lru_cache_get_missing_key_returns_none():
    cache = _LRUCache()
    assert cache.get("missing") is None


def test_lru_cache_set_then_get_roundtrips():
    cache = _LRUCache()
    cache.set("k", {"answer": "hi"}, ttl_seconds=60)
    assert cache.get("k") == {"answer": "hi"}


def test_lru_cache_expired_entry_returns_none():
    cache = _LRUCache()
    cache.set("k", "value", ttl_seconds=-1)  # already expired
    assert cache.get("k") is None


def test_lru_cache_evicts_oldest_when_over_capacity():
    cache = _LRUCache(max_size=2)
    cache.set("a", 1, ttl_seconds=60)
    cache.set("b", 2, ttl_seconds=60)
    cache.set("c", 3, ttl_seconds=60)

    assert cache.get("a") is None  # evicted, oldest
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_delete_removes_key():
    cache = _LRUCache()
    cache.set("k", "value", ttl_seconds=60)
    cache.delete("k")
    assert cache.get("k") is None


# --- ResponseCache.make_key --------------------------------------------------


def test_make_key_normalizes_whitespace_and_case():
    key1 = ResponseCache.make_key("rag", "What is the Refund   Policy?")
    key2 = ResponseCache.make_key("rag", "what is the refund policy?")
    assert key1 == key2


def test_make_key_different_namespace_differs():
    key1 = ResponseCache.make_key("rag", "same query")
    key2 = ResponseCache.make_key("sql", "same query")
    assert key1 != key2


# --- Redis unavailable: falls back to LRU without crashing ------------------


def test_cache_falls_back_to_lru_when_no_redis_url():
    cache = ResponseCache(redis_url=None)
    assert cache._redis is None

    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_falls_back_to_lru_when_redis_connection_fails():
    """The reported degrade-gracefully contract: a Redis connect/ping
    failure at construction must not raise -- just silently fall back
    to the in-process LRU.
    """

    with patch("redis.Redis.from_url", side_effect=ConnectionError("connection refused")):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    assert cache._redis is None

    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_falls_back_to_lru_when_ping_fails():
    mock_client = MagicMock()
    mock_client.ping.side_effect = ConnectionError("no route to host")

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    assert cache._redis is None


# --- Redis available (mocked): cache actually uses it ------------------------


def test_cache_uses_redis_when_available():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    assert cache._redis is mock_client

    cache.set("k", {"answer": "hi"}, ttl_seconds=120)
    mock_client.set.assert_called_once()
    args, kwargs = mock_client.set.call_args
    assert args[0] == "k"
    assert '"answer": "hi"' in args[1]
    assert kwargs["ex"] == 120


def test_cache_get_deserializes_redis_value():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = '{"answer": "cached"}'

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    assert cache.get("k") == {"answer": "cached"}


def test_cache_invalidate_calls_redis_delete_and_clears_lru():
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    cache._lru.set("k", "v", ttl_seconds=60)  # simulate a stale local copy
    cache.invalidate("k")

    mock_client.delete.assert_called_once_with("k")
    assert cache._lru.get("k") is None


# --- Redis fails mid-operation: falls back to LRU without crashing ----------


def test_cache_get_falls_back_to_lru_when_redis_get_fails():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.side_effect = ConnectionError("connection reset")

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    cache._lru.set("k", "fallback-value", ttl_seconds=60)
    assert cache.get("k") == "fallback-value"


def test_cache_set_falls_back_to_lru_when_redis_set_fails():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.set.side_effect = ConnectionError("connection reset")

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    cache.set("k", "v")  # must not raise
    assert cache._lru.get("k") == "v"


def test_cache_invalidate_does_not_raise_when_redis_delete_fails():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.delete.side_effect = ConnectionError("connection reset")

    with patch("redis.Redis.from_url", return_value=mock_client):
        cache = ResponseCache(redis_url="redis://localhost:6379/0")

    cache.invalidate("k")  # must not raise

"""Redis-backed response cache with an in-process LRU fallback.

Caches by a normalized-query hash so repeated questions (common in
demos/evals and duplicate user queries) skip the full retrieval +
generation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("cache")

DEFAULT_TTL_SECONDS = 3600


class _LRUCache:
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at < time.monotonic():
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._store[key] = (value, time.monotonic() + ttl_seconds)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class ResponseCache:
    """Tries Redis first (shared across processes); falls back to an
    in-memory LRU if Redis is unavailable, so the app degrades gracefully
    instead of failing when the cache dependency is down.
    """

    def __init__(self, redis_url: str | None = None, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_lru_size: int = 512):
        self.ttl_seconds = ttl_seconds
        self._lru = _LRUCache(max_size=max_lru_size)
        self._redis = None

        redis_url = redis_url or os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=1)
                self._redis.ping()
                logger.info("connected to redis cache at %s", redis_url)
            except Exception as exc:
                logger.warning("redis unavailable (%s), falling back to in-memory cache", exc)
                self._redis = None

    @staticmethod
    def make_key(namespace: str, query: str) -> str:
        normalized = " ".join(query.strip().lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception as exc:
                logger.warning("redis get failed (%s), falling back to lru", exc)
        return self._lru.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self.ttl_seconds
        if self._redis is not None:
            try:
                self._redis.set(key, json.dumps(value), ex=ttl)
                return
            except Exception as exc:
                logger.warning("redis set failed (%s), falling back to lru", exc)
        self._lru.set(key, value, ttl)

    def invalidate(self, key: str) -> None:
        if self._redis is not None:
            try:
                self._redis.delete(key)
            except Exception as exc:
                logger.warning("redis delete failed (%s)", exc)
        self._lru.delete(key)

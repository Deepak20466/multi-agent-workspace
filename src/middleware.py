"""HTTP-layer middleware for the FastAPI app: API-key auth and a Redis
token-bucket rate limiter (60 requests/min per user_id by default).

Both degrade gracefully the same way src/cache.py does: if Redis is
unreachable, the rate limiter fails open (allows the request) rather
than taking the whole API down over a cache-tier outage.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Atomic token-bucket refill + consume, keyed per caller. Runs entirely
# in Redis so concurrent requests across processes can't race past the
# limit between a GET and a SET.
_TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call("HMGET", key, "tokens", "timestamp")
local tokens = tonumber(bucket[1])
local timestamp = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    timestamp = now
end

local delta = math.max(0, now - timestamp)
tokens = math.min(capacity, tokens + delta * refill_rate)

local allowed = 0
if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
end

redis.call("HMSET", key, "tokens", tokens, "timestamp", now)
redis.call("EXPIRE", key, 120)

return allowed
"""


class RateLimiter:
    """Redis-backed token bucket: `capacity` requests, refilling at
    `capacity` per `window_seconds`. Construct once per process and
    share it across requests (holds a pooled Redis connection).
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        capacity: int = 60,
        window_seconds: float = 60.0,
    ):
        self.capacity = capacity
        self.refill_rate = capacity / window_seconds
        self._redis = None
        self._script = None

        redis_url = redis_url or os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, socket_connect_timeout=1)
                self._script = self._redis.register_script(_TOKEN_BUCKET_SCRIPT)
            except Exception as exc:
                logger.warning("rate limiter: redis unavailable ({}), failing open", exc)
                self._redis = None

    async def allow(self, key: str) -> bool:
        if self._redis is None or self._script is None:
            return True
        try:
            allowed = await self._script(
                keys=[f"ratelimit:{key}"],
                args=[self.capacity, self.refill_rate, time.time()],
            )
            return bool(int(allowed))
        except Exception as exc:
            logger.warning("rate limiter: redis call failed ({}), failing open", exc)
            return True


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Requires `X-API-Key` to match the `API_KEY` env var on every
    request except the exempt paths below. If `API_KEY` isn't set, auth
    is skipped entirely (local/dev default) so a fresh checkout isn't
    locked out with no key configured.
    """

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        api_key = os.getenv("API_KEY")
        if api_key and request.url.path not in self.EXEMPT_PATHS:
            if request.headers.get("X-API-Key") != api_key:
                return JSONResponse({"detail": "invalid or missing X-API-Key"}, status_code=401)
        return await call_next(request)

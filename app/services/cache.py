from __future__ import annotations
import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, Optional

try:
    from redis.asyncio import Redis
    from redis.asyncio.connection import ConnectionPool
except Exception:  # pragma: no cover
    Redis = None
    ConnectionPool = None

from app.core.config import settings


class SearchCache:
    def __init__(self) -> None:
        self._enabled = settings.redis_enabled
        self._ttl = settings.redis_ttl_sec
        self._client: Optional[Redis] = None
        self._namespace = settings.redis_namespace
        self._hit_count = 0
        self._miss_count = 0
        self._set_count = 0
        self._delete_count = 0
        self._lock_contention_count = 0
        if self._enabled and Redis is not None:
            if ConnectionPool is not None:
                pool = ConnectionPool.from_url(
                    settings.redis_url,
                    max_connections=settings.redis_max_connections,
                    socket_timeout=settings.redis_socket_timeout_sec,
                    decode_responses=True,
                )
                self._client = Redis(connection_pool=pool)
            else:
                self._client = Redis.from_url(settings.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def health(self) -> str:
        if not self._enabled:
            return "disabled"
        try:
            assert self._client is not None
            pong = await self._client.ping()
            return "up" if pong else "degraded"
        except Exception:
            return "down"

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"search:{digest}"

    def _ns(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    async def get_json(self, key: str) -> Optional[dict]:
        if not self._enabled:
            return None
        try:
            assert self._client is not None
            value = await self._client.get(self._ns(key))
            if not value:
                self._miss_count += 1
                return None
            self._hit_count += 1
            return json.loads(value)
        except Exception:
            return None

    async def set_json(self, key: str, payload: dict) -> None:
        if not self._enabled:
            return
        try:
            assert self._client is not None
            await self._client.set(self._ns(key), json.dumps(payload, ensure_ascii=True), ex=self._ttl)
            self._set_count += 1
        except Exception:
            return

    async def acquire_lock(self, key: str, owner: Optional[str] = None) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            assert self._client is not None
            token = owner or str(uuid.uuid4())
            lock_key = self._ns(f"lock:{key}")
            ok = await self._client.set(lock_key, token, nx=True, ex=settings.redis_lock_ttl_sec)
            if ok:
                return token
            self._lock_contention_count += 1
            return None
        except Exception:
            return None

    async def wait_for_value(self, key: str) -> Optional[dict]:
        if not self._enabled:
            return None
        wait_deadline = time.time() + (settings.redis_lock_wait_ms / 1000.0)
        while time.time() < wait_deadline:
            cached = await self.get_json(key)
            if cached is not None:
                return cached
            await asyncio.sleep(0.03)
        return None

    async def release_lock(self, key: str, owner: str) -> None:
        if not self._enabled:
            return
        try:
            assert self._client is not None
            lock_key = self._ns(f"lock:{key}")
            current = await self._client.get(lock_key)
            if current == owner:
                await self._client.delete(lock_key)
        except Exception:
            return

    async def invalidate_by_prefix(self, prefix: str) -> int:
        if not self._enabled:
            return 0
        try:
            assert self._client is not None
            pattern = self._ns(prefix) + "*"
            deleted = 0
            async for key in self._client.scan_iter(match=pattern, count=1000):
                deleted += await self._client.delete(key)
            self._delete_count += deleted
            return int(deleted)
        except Exception:
            return 0

    async def key_count(self, prefix: str = "search:") -> int:
        if not self._enabled:
            return 0
        try:
            assert self._client is not None
            pattern = self._ns(prefix) + "*"
            count = 0
            async for _ in self._client.scan_iter(match=pattern, count=1000):
                count += 1
            return count
        except Exception:
            return 0

    async def stats(self) -> dict:
        status = await self.health()
        keys = await self.key_count("search:")
        return {
            "status": status,
            "key_count": keys,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "set_count": self._set_count,
            "delete_count": self._delete_count,
            "lock_contention_count": self._lock_contention_count,
        }





import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logging import logger


class RedisService:
    """
    Unified Redis service covering the 6 core architectural responsibilities:
    1. Ingestion buffer (Redis Streams)
    2. API response cache (short TTL ~10-30s)
    3. FRIDAY session memory
    4. Rate-limit counters & distributed job locks (SETNX with TTL)
    5. LLM circuit-breaker / cooldown state
    6. JWT revocation blacklist

    Includes resilient in-memory fallback for seamless local development
    when a standalone Redis daemon is not running.
    """

    def __init__(self) -> None:
        self.redis: Optional[aioredis.Redis] = None
        # In-memory storage structures for offline fallback
        self._memory_store: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._memory_streams: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self._memory_session: Dict[str, Tuple[List[Dict[str, Any]], Optional[float]]] = {}
        self._memory_locks: Dict[str, float] = {}
        self._memory_cooldowns: Dict[str, float] = {}

    def _clean_expired(self) -> None:
        """Purges expired in-memory cache, session, and lock entries."""
        now = time.time()
        for k in [k for k, (_, exp) in self._memory_store.items() if exp and exp < now]:
            self._memory_store.pop(k, None)
        for k in [k for k, (_, exp) in self._memory_session.items() if exp and exp < now]:
            self._memory_session.pop(k, None)
        for k in [k for k, exp in self._memory_locks.items() if exp and exp < now]:
            self._memory_locks.pop(k, None)
        for k in [k for k, exp in self._memory_cooldowns.items() if exp and exp < now]:
            self._memory_cooldowns.pop(k, None)

    async def connect(self) -> None:
        try:
            self.redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info("Connected to Redis successfully on %s", settings.REDIS_URL)
        except Exception as e:
            logger.info("Redis server offline (%s); operating with in-memory fallback cache & queue", e)
            self.redis = None

    async def disconnect(self) -> None:
        if self.redis:
            await self.redis.close()
            logger.info("Closed Redis connection")

    # -------------------------------------------------------------
    # 1. Ingestion Buffer (Redis Streams)
    # -------------------------------------------------------------
    async def add_to_stream(self, stream_key: str, data: Dict[str, Any]) -> Optional[str]:
        if not self.redis:
            # In-memory stream queue fallback
            entry_id = f"{int(time.time() * 1000)}-{len(self._memory_streams.get(stream_key, []))}"
            if stream_key not in self._memory_streams:
                self._memory_streams[stream_key] = []
            self._memory_streams[stream_key].append((entry_id, data))
            # Keep in-memory buffer bounded to last 1000 entries
            if len(self._memory_streams[stream_key]) > 1000:
                self._memory_streams[stream_key] = self._memory_streams[stream_key][-1000:]
            return entry_id

        try:
            serialized = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()}
            entry_id = await self.redis.xadd(stream_key, serialized)
            return entry_id
        except Exception as e:
            logger.error("Error adding to Redis Stream %s: %s", stream_key, e, exc_info=True)
            return None

    async def check_idempotency_key(self, key: str, ttl_seconds: int = 86400) -> bool:
        """
        Returns True if this key is NEW and was successfully acquired.
        Returns False if the key already exists (duplicate event).
        """
        if not self.redis:
            self._clean_expired()
            store_key = f"idempotency:{key}"
            if store_key in self._memory_store:
                return False
            self._memory_store[store_key] = ("1", time.time() + ttl_seconds)
            return True

        try:
            is_new = await self.redis.set(f"idempotency:{key}", "1", nx=True, ex=ttl_seconds)
            return bool(is_new)
        except Exception as e:
            logger.error("Error checking idempotency for key %s: %s", key, e, exc_info=True)
            return True

    # -------------------------------------------------------------
    # 2. API Response Cache
    # -------------------------------------------------------------
    async def get_cache(self, key: str) -> Optional[Any]:
        if not self.redis:
            self._clean_expired()
            store_key = f"cache:{key}"
            entry = self._memory_store.get(store_key)
            return entry[0] if entry else None

        try:
            val = await self.redis.get(f"cache:{key}")
            return json.loads(val) if val else None
        except Exception as e:
            logger.error("Error fetching cache for %s: %s", key, e, exc_info=True)
            return None

    async def set_cache(self, key: str, value: Any, ttl_seconds: int = 15) -> bool:
        if not self.redis:
            store_key = f"cache:{key}"
            self._memory_store[store_key] = (value, time.time() + ttl_seconds)
            return True

        try:
            val = json.dumps(value)
            await self.redis.set(f"cache:{key}", val, ex=ttl_seconds)
            return True
        except Exception as e:
            logger.error("Error setting cache for %s: %s", key, e, exc_info=True)
            return False

    async def invalidate_cache_pattern(self, pattern: str) -> None:
        if not self.redis:
            clean_pattern = pattern.replace("*", "")
            for k in list(self._memory_store.keys()):
                if clean_pattern in k:
                    self._memory_store.pop(k, None)
            return

        try:
            keys = await self.redis.keys(f"cache:{pattern}")
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.error("Error invalidating cache pattern %s: %s", pattern, e, exc_info=True)

    # -------------------------------------------------------------
    # 3. FRIDAY Session Memory
    # -------------------------------------------------------------
    async def get_session_memory(self, business_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        if not self.redis:
            self._clean_expired()
            key = f"session:{business_id}:{conversation_id}"
            entry = self._memory_session.get(key)
            return list(entry[0]) if entry else []

        try:
            key = f"session:{business_id}:{conversation_id}"
            raw = await self.redis.get(key)
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.error("Error reading session memory for %s/%s: %s", business_id, conversation_id, e, exc_info=True)
            return []

    async def save_session_memory(
        self,
        business_id: str,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        ttl_seconds: int = 86400,
    ) -> None:
        if not self.redis:
            key = f"session:{business_id}:{conversation_id}"
            self._memory_session[key] = (list(messages), time.time() + ttl_seconds)
            return

        try:
            key = f"session:{business_id}:{conversation_id}"
            await self.redis.set(key, json.dumps(messages), ex=ttl_seconds)
        except Exception as e:
            logger.error("Error saving session memory for %s/%s: %s", business_id, conversation_id, e, exc_info=True)

    # -------------------------------------------------------------
    # 4. Distributed Job Locks & Rate-Limiting
    # -------------------------------------------------------------
    async def acquire_lock(self, lock_name: str, ttl_seconds: int = 60) -> bool:
        """Acquires a distributed lock using SETNX with TTL."""
        if not self.redis:
            self._clean_expired()
            now = time.time()
            if lock_name in self._memory_locks and self._memory_locks[lock_name] > now:
                return False
            self._memory_locks[lock_name] = now + ttl_seconds
            return True

        try:
            acquired = await self.redis.set(f"lock:{lock_name}", "locked", nx=True, ex=ttl_seconds)
            return bool(acquired)
        except Exception as e:
            logger.error("Error acquiring lock %s: %s", lock_name, e, exc_info=True)
            return True

    async def release_lock(self, lock_name: str) -> None:
        if not self.redis:
            self._memory_locks.pop(lock_name, None)
            return

        try:
            await self.redis.delete(f"lock:{lock_name}")
        except Exception as e:
            logger.error("Error releasing lock %s: %s", lock_name, e, exc_info=True)

    # -------------------------------------------------------------
    # 5. LLM Circuit-Breaker / Cooldown State
    # -------------------------------------------------------------
    async def set_llm_cooldown(self, endpoint: str, duration_seconds: int) -> None:
        if not self.redis:
            self._memory_cooldowns[endpoint] = time.time() + duration_seconds
            return

        try:
            key = f"llm:cooldown:{endpoint}"
            await self.redis.set(key, "cooling_down", ex=duration_seconds)
        except Exception as e:
            logger.error("Error setting LLM cooldown for %s: %s", endpoint, e, exc_info=True)

    async def is_llm_cooling_down(self, endpoint: str) -> bool:
        if not self.redis:
            self._clean_expired()
            return endpoint in self._memory_cooldowns and self._memory_cooldowns[endpoint] > time.time()

        try:
            return bool(await self.redis.exists(f"llm:cooldown:{endpoint}"))
        except Exception as e:
            logger.error("Error checking LLM cooldown for %s: %s", endpoint, e, exc_info=True)
            return False

    async def clear_llm_cooldown(self, endpoint: str) -> None:
        if not self.redis:
            self._memory_cooldowns.pop(endpoint, None)
            return

        try:
            await self.redis.delete(f"llm:cooldown:{endpoint}")
        except Exception as e:
            logger.error("Error clearing LLM cooldown for %s: %s", endpoint, e, exc_info=True)

    # -------------------------------------------------------------
    # 6. JWT Revocation Blacklist
    # -------------------------------------------------------------
    async def revoke_token(self, jti: str, ttl_seconds: int) -> None:
        """Blacklist a revoked JWT until its expiry time."""
        if not self.redis:
            self._memory_store[f"jwt:blacklist:{jti}"] = ("revoked", time.time() + ttl_seconds)
            return

        try:
            await self.redis.set(f"jwt:blacklist:{jti}", "revoked", ex=ttl_seconds)
        except Exception as e:
            logger.error("Error blacklisting token %s: %s", jti, e, exc_info=True)

    async def is_token_revoked(self, jti: str) -> bool:
        if not self.redis:
            self._clean_expired()
            return f"jwt:blacklist:{jti}" in self._memory_store

        try:
            return bool(await self.redis.exists(f"jwt:blacklist:{jti}"))
        except Exception as e:
            logger.error("Error checking token blacklist for %s: %s", jti, e, exc_info=True)
            return False


# Global singleton instance
redis_service = RedisService()

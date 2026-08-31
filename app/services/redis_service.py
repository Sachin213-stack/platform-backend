import json
import asyncio
from typing import Any, Dict, List, Optional
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
    """

    def __init__(self) -> None:
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        try:
            self.redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed (running in fallback/mock mode): {e}")
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
            return None
        try:
            # Flatten or serialize nested dicts to JSON strings for stream entry
            serialized = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()}
            entry_id = await self.redis.xadd(stream_key, serialized)
            return entry_id
        except Exception as e:
            logger.error(f"Error adding to Redis Stream {stream_key}: {e}")
            return None

    async def check_idempotency_key(self, key: str, ttl_seconds: int = 86400) -> bool:
        """
        Returns True if this key is NEW and was successfully acquired.
        Returns False if the key already exists (duplicate event).
        """
        if not self.redis:
            return True
        try:
            is_new = await self.redis.set(f"idempotency:{key}", "1", nx=True, ex=ttl_seconds)
            return bool(is_new)
        except Exception as e:
            logger.error(f"Error checking idempotency for key {key}: {e}")
            return True

    # -------------------------------------------------------------
    # 2. API Response Cache
    # -------------------------------------------------------------
    async def get_cache(self, key: str) -> Optional[Any]:
        if not self.redis:
            return None
        try:
            val = await self.redis.get(f"cache:{key}")
            return json.loads(val) if val else None
        except Exception as e:
            logger.error(f"Error fetching cache for {key}: {e}")
            return None

    async def set_cache(self, key: str, value: Any, ttl_seconds: int = 15) -> bool:
        if not self.redis:
            return False
        try:
            val = json.dumps(value)
            await self.redis.set(f"cache:{key}", val, ex=ttl_seconds)
            return True
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {e}")
            return False

    async def invalidate_cache_pattern(self, pattern: str) -> None:
        if not self.redis:
            return
        try:
            keys = await self.redis.keys(f"cache:{pattern}")
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Error invalidating cache pattern {pattern}: {e}")

    # -------------------------------------------------------------
    # 3. FRIDAY Session Memory
    # -------------------------------------------------------------
    async def get_session_memory(self, business_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        if not self.redis:
            return []
        try:
            key = f"session:{business_id}:{conversation_id}"
            raw = await self.redis.get(key)
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.error(f"Error reading session memory: {e}")
            return []

    async def save_session_memory(
        self,
        business_id: str,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        ttl_seconds: int = 86400,
    ) -> None:
        if not self.redis:
            return
        try:
            key = f"session:{business_id}:{conversation_id}"
            await self.redis.set(key, json.dumps(messages), ex=ttl_seconds)
        except Exception as e:
            logger.error(f"Error saving session memory: {e}")

    # -------------------------------------------------------------
    # 4. Distributed Job Locks & Rate-Limiting
    # -------------------------------------------------------------
    async def acquire_lock(self, lock_name: str, ttl_seconds: int = 60) -> bool:
        """Acquires a distributed lock using SETNX with TTL."""
        if not self.redis:
            return True
        try:
            acquired = await self.redis.set(f"lock:{lock_name}", "locked", nx=True, ex=ttl_seconds)
            return bool(acquired)
        except Exception as e:
            logger.error(f"Error acquiring lock {lock_name}: {e}")
            return True

    async def release_lock(self, lock_name: str) -> None:
        if not self.redis:
            return
        try:
            await self.redis.delete(f"lock:{lock_name}")
        except Exception as e:
            logger.error(f"Error releasing lock {lock_name}: {e}")

    # -------------------------------------------------------------
    # 5. LLM Circuit-Breaker / Cooldown State
    # -------------------------------------------------------------
    async def set_llm_cooldown(self, endpoint: str, duration_seconds: int) -> None:
        if not self.redis:
            return
        try:
            key = f"llm:cooldown:{endpoint}"
            await self.redis.set(key, "cooling_down", ex=duration_seconds)
        except Exception as e:
            logger.error(f"Error setting LLM cooldown: {e}")

    async def is_llm_cooling_down(self, endpoint: str) -> bool:
        if not self.redis:
            return False
        try:
            return bool(await self.redis.exists(f"llm:cooldown:{endpoint}"))
        except Exception:
            return False

    async def clear_llm_cooldown(self, endpoint: str) -> None:
        if not self.redis:
            return
        try:
            await self.redis.delete(f"llm:cooldown:{endpoint}")
        except Exception as e:
            logger.error(f"Error clearing LLM cooldown: {e}")

    # -------------------------------------------------------------
    # 6. JWT Revocation Blacklist
    # -------------------------------------------------------------
    async def revoke_token(self, jti: str, ttl_seconds: int) -> None:
        """Blacklist a revoked JWT until its expiry time."""
        if not self.redis:
            return
        try:
            await self.redis.set(f"jwt:blacklist:{jti}", "revoked", ex=ttl_seconds)
        except Exception as e:
            logger.error(f"Error blacklisting token {jti}: {e}")

    async def is_token_revoked(self, jti: str) -> bool:
        if not self.redis:
            return False
        try:
            return bool(await self.redis.exists(f"jwt:blacklist:{jti}"))
        except Exception:
            return False


# Global singleton instance
redis_service = RedisService()

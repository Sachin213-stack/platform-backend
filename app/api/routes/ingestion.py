import re
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, status
from sqlalchemy import select

from app.core.config import settings
from app.core.logging import logger
from app.api.schemas.ingestion import TelemetryEventCreate, IngestionResponse
from app.core.security import decode_token
from app.services.redis_service import redis_service
from app.db.session import AsyncSessionLocal, is_db_available
from app.db.models.business import ApiKey

router = APIRouter(prefix="/ingestion", tags=["Ingestion Pipeline"])


def _sanitize_endpoint(raw_endpoint: Optional[str]) -> str:
    """Sanitizes endpoint strings to prevent prompt injection and header smuggling."""
    if not raw_endpoint:
        return ""
    # Strip newlines, control characters, and tabs
    cleaned = re.sub(r"[\r\n\x00-\x1f\x7f]", "", raw_endpoint).strip()
    # Strip potential LLM prompt injection delimiters, XML tags, and overrides
    cleaned = re.sub(
        r"(?i)(\[system|\<\||\<system|<\s*/?\s*untrusted_.*?/?>|system override|disregard previous instructions|ignore previous instructions)",
        "",
        cleaned,
    )
    # Neutralize angle brackets and backticks in endpoints
    cleaned = cleaned.replace("<", "").replace(">", "").replace("`", "")
    return cleaned[:255]


def _sanitize_metadata(data: Any, max_depth: int = 4) -> Any:
    """Recursively sanitizes telemetry payload metadata to strip control characters and prompt injections."""
    if max_depth <= 0 or data is None:
        return {} if isinstance(data, dict) else ""

    if isinstance(data, str):
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", data).strip()
        cleaned = re.sub(r"(?i)(<\s*/?\s*untrusted_.*?/?>|\[SYSTEM\s+OVERRIDE\]|<\|im_start\||<\|im_end\|)", "", cleaned)
        return cleaned[:500]
    elif isinstance(data, dict):
        return {
            str(k)[:64]: _sanitize_metadata(v, max_depth - 1)
            for k, v in list(data.items())[:30]
        }
    elif isinstance(data, list):
        return [_sanitize_metadata(item, max_depth - 1) for item in data[:30]]
    elif isinstance(data, (int, float, bool)):
        return data
    return str(data)[:100]


async def _resolve_api_key_tenant(raw_key: str) -> Optional[str]:
    """Resolves an API key to its associated business_id via Redis cache or Postgres."""
    if not raw_key:
        return None
    raw_key = raw_key.strip()

    # Check Redis cache first
    cache_key = f"apikey:resolved:{raw_key[:16]}"
    cached_biz_id = await redis_service.get_cache(cache_key)
    if cached_biz_id:
        return str(cached_biz_id)

    # Allow local development test key
    if settings.ENVIRONMENT == "development" and raw_key in ["aicto_dev_telemetry_key", "test-api-key"]:
        return "11111111-1111-1111-1111-111111111111"

    # Query DB if available
    if await is_db_available():
        try:
            prefix = raw_key[:12] + "..."
            async with AsyncSessionLocal() as session:
                stmt = select(ApiKey).where(ApiKey.key_prefix == prefix, ApiKey.is_active.is_(True))
                keys = (await session.execute(stmt)).scalars().all()
                for key_obj in keys:
                    if key_obj.get_key() == raw_key:
                        biz_id_str = str(key_obj.business_id)
                        await redis_service.set_cache(cache_key, biz_id_str, ttl_seconds=300)
                        return biz_id_str
        except Exception as e:
            logger.warning("Could not verify API key against database: %s", e)

    return None


@router.post("/events", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry_event(
    event: TelemetryEventCreate,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Zero-trust telemetry ingestion path:
    1. Strictly authenticates via Bearer JWT token or X-API-Key header.
    2. Enforces that event payload business_id matches the authenticated tenant.
    3. Sanitizes endpoint strings and metadata to prevent injection attacks.
    4. Enqueues event payload to persistent Redis Stream.
    """
    authenticated_biz_id: Optional[str] = None

    # 1. Check Bearer Token
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()
        payload = decode_token(token)
        if payload and payload.get("type") != "refresh":
            jti = payload.get("jti")
            if not (jti and await redis_service.is_token_revoked(jti)):
                authenticated_biz_id = payload.get("business_id")

    # 2. Check X-API-Key Header if Bearer token not provided or invalid
    if not authenticated_biz_id and x_api_key:
        authenticated_biz_id = await _resolve_api_key_tenant(x_api_key)

    # 3. Reject unauthenticated requests
    if not authenticated_biz_id:
        logger.warning("Rejected unauthenticated telemetry ingestion attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Prevent cross-tenant spoofing: if payload specifies a different business_id, reject
    if event.business_id and str(event.business_id) != str(authenticated_biz_id):
        logger.warning(
            "Cross-tenant poisoning attempt blocked: Authenticated tenant '%s' tried to submit event for tenant '%s'",
            authenticated_biz_id,
            event.business_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Payload business_id does not match authenticated credentials.",
        )

    business_id = authenticated_biz_id

    # 5. Check Idempotency Key
    if event.idempotency_key:
        is_new = await redis_service.check_idempotency_key(event.idempotency_key)
        if not is_new:
            logger.debug("Duplicate event skipped (idempotency key: %s)", event.idempotency_key)
            return IngestionResponse(
                status="skipped",
                message="Duplicate event ignored",
                duplicate=True,
            )

    # 6. Prepare sanitized payload for Redis Stream with Clamping Flags
    clean_endpoint = _sanitize_endpoint(event.endpoint)

    raw_rt = float(event.response_time_ms or 0.0)
    raw_cpu = float(event.cpu_usage_pct or 0.0)
    raw_mem = float(event.memory_usage_pct or 0.0)

    is_rt_clamped = raw_rt > 60000.0 or raw_rt < 0.0
    is_cpu_clamped = raw_cpu > 100.0 or raw_cpu < 0.0
    is_mem_clamped = raw_mem > 100.0 or raw_mem < 0.0

    metadata = _sanitize_metadata(event.payload_metadata) if isinstance(event.payload_metadata, dict) else {}
    if is_rt_clamped or is_cpu_clamped or is_mem_clamped:
        metadata["clamping"] = {
            "is_clamped": True,
            "raw_response_time_ms": raw_rt,
            "raw_cpu_usage_pct": raw_cpu,
            "raw_memory_usage_pct": raw_mem,
            "reason": "Exceeded safe operational numerical boundaries",
        }

    event_payload = {
        "id": str(uuid.uuid4()),
        "business_id": business_id,
        "idempotency_key": event.idempotency_key or "",
        "event_type": event.event_type,
        "response_time_ms": max(0.0, min(60000.0, raw_rt)),
        "status_code": max(100, min(599, int(event.status_code or 200))),
        "orders_count": max(0, int(event.orders_count or 0)),
        "revenue_amount": max(0.0, float(event.revenue_amount or 0.0)),
        "cpu_usage_pct": max(0.0, min(100.0, raw_cpu)),
        "memory_usage_pct": max(0.0, min(100.0, raw_mem)),
        "queue_depth": max(0, int(event.queue_depth or 0)),
        "endpoint": clean_endpoint,
        "payload_metadata": metadata,
        "timestamp": (event.timestamp or datetime.now(timezone.utc)).isoformat(),
    }

    # 7. Add to Redis Stream
    entry_id = await redis_service.add_to_stream(
        stream_key=settings.REDIS_STREAM_KEY,
        data=event_payload,
    )
    logger.debug(
        "Telemetry event queued to stream %s (id: %s, tenant: %s)",
        settings.REDIS_STREAM_KEY,
        event_payload["id"],
        business_id,
    )

    return IngestionResponse(
        status="accepted",
        message="Telemetry event queued for batch processing",
        event_id=entry_id or event_payload["id"],
        duplicate=False,
    )


@router.get("/dlq")
async def get_dlq_events(
    limit: int = 50,
    authorization: Optional[str] = Header(None),
):
    """
    Inspects quarantined Dead-Letter Queue (DLQ) telemetry events.
    Requires Bearer token authentication with admin or owner privileges.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to inspect Dead-Letter Queue.",
        )
    token = authorization.replace("Bearer ", "").strip()
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")

    user_role = payload.get("role", "viewer")
    if user_role not in ["owner", "admin"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or owner can inspect DLQ.")

    dlq_events = []
    if redis_service.redis:
        try:
            entries = await redis_service.redis.xrevrange(
                settings.REDIS_DLQ_STREAM_KEY,
                count=min(100, max(1, limit)),
            )
            for entry_id, fields in entries:
                dlq_events.append({
                    "entry_id": entry_id if isinstance(entry_id, str) else entry_id.decode(),
                    "data": {
                        (k if isinstance(k, str) else k.decode()): (v if isinstance(v, str) else v.decode())
                        for k, v in fields.items()
                    }
                })
        except Exception as e:
            logger.error("Failed to read from Redis DLQ stream: %s", e)

    return {
        "dlq_stream": settings.REDIS_DLQ_STREAM_KEY,
        "total_quarantined": len(dlq_events),
        "events": dlq_events,
    }

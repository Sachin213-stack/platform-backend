import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Header, status
from app.core.config import settings
from app.core.logging import logger
from app.api.schemas.ingestion import TelemetryEventCreate, IngestionResponse
from app.core.security import decode_token
from app.services.redis_service import redis_service

router = APIRouter(prefix="/ingestion", tags=["Ingestion Pipeline"])


@router.post("/events", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry_event(
    event: TelemetryEventCreate,
    authorization: Optional[str] = Header(None),
    x_business_id: Optional[str] = Header(None),
):
    """
    High-throughput ingestion hot path:
    Supports authenticated API calls and client-side beacon snippets.
    1. Verify client UUID idempotency key via Redis SETNX to discard duplicates.
    2. Enqueue event payload to persistent Redis Stream.
    3. Return 202 Accepted immediately.
    """
    business_id = event.business_id or x_business_id or "default-tenant"
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        payload = decode_token(token)
        if payload and "business_id" in payload:
            business_id = payload["business_id"]

    # 1. Check Idempotency Key
    if event.idempotency_key:
        is_new = await redis_service.check_idempotency_key(event.idempotency_key)
        if not is_new:
            logger.debug("Duplicate event skipped (idempotency key: %s)", event.idempotency_key)
            return IngestionResponse(
                status="skipped",
                message="Duplicate event ignored",
                duplicate=True,
            )

    # 2. Prepare payload for Redis Stream
    event_payload = {
        "id": str(uuid.uuid4()),
        "business_id": business_id,
        "idempotency_key": event.idempotency_key or "",
        "event_type": event.event_type,
        "response_time_ms": event.response_time_ms or 0.0,
        "status_code": event.status_code or 200,
        "orders_count": event.orders_count or 0,
        "revenue_amount": event.revenue_amount or 0.0,
        "cpu_usage_pct": event.cpu_usage_pct or 0.0,
        "memory_usage_pct": event.memory_usage_pct or 0.0,
        "queue_depth": event.queue_depth or 0,
        "endpoint": event.endpoint or "",
        "payload_metadata": event.payload_metadata or {},
        "timestamp": (event.timestamp or datetime.now(timezone.utc)).isoformat(),
    }

    # 3. Add to Redis Stream
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

import time
import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db, is_db_available
from app.services.redis_service import redis_service
from app.api.schemas.health import HealthResponse, ServiceStatus

router = APIRouter(prefix="/health", tags=["Observability"])


@router.get("", response_model=HealthResponse)
async def check_health(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Comprehensive diagnostic health probe checking:
    1. PostgreSQL connection & read latency
    2. Redis connection & ping latency
    3. Kimi (Moonshot AI) upstream accessibility
    4. Background Workers (MLWorker & IngestionWorker)
    """
    services = {}
    overall_healthy = True

    # 1. Check PostgreSQL
    db_available = await is_db_available()
    if db_available:
        db_start = time.perf_counter()
        try:
            await db.execute(text("SELECT 1"))
            db_latency = (time.perf_counter() - db_start) * 1000
            services["postgresql"] = ServiceStatus(
                status="healthy",
                latency_ms=round(db_latency, 2),
                message="Database query executed successfully",
            )
        except Exception as e:
            overall_healthy = False
            logger.warning("Health probe PostgreSQL connection lost: %s", e)
            services["postgresql"] = ServiceStatus(
                status="unhealthy",
                message=f"PostgreSQL connection failed: {str(e)[:100]}",
            )
    else:
        overall_healthy = False
        logger.info("Health probe PostgreSQL: offline (decoupled dev mode active)")
        services["postgresql"] = ServiceStatus(
            status="degraded",
            message="PostgreSQL offline (decoupled dev mode active)",
        )

    # 2. Check Redis
    redis_start = time.perf_counter()
    if redis_service.redis:
        try:
            await redis_service.redis.ping()
            redis_latency = (time.perf_counter() - redis_start) * 1000
            services["redis"] = ServiceStatus(
                status="healthy",
                latency_ms=round(redis_latency, 2),
                message="Redis ping successful",
            )
        except Exception as e:
            overall_healthy = False
            logger.warning("Health probe Redis ping failed: %s", e)
            services["redis"] = ServiceStatus(
                status="unhealthy",
                message=f"Redis ping failed: {str(e)[:100]}",
            )
    else:
        logger.debug("Health probe Redis: in-memory fallback active")
        services["redis"] = ServiceStatus(
            status="degraded",
            message="Redis client not connected (running in in-memory fallback)",
        )

    # 3. Check Kimi (Moonshot AI) connectivity
    kimi_key = settings.effective_kimi_api_key
    if kimi_key:
        kimi_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(
                    f"{settings.KIMI_BASE_URL.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {kimi_key}"},
                )
                kimi_latency = (time.perf_counter() - kimi_start) * 1000
                status_str = "healthy" if res.status_code == 200 else "degraded"
                if status_str != "healthy":
                    logger.warning("Health probe Kimi returned non-200 status: HTTP %d", res.status_code)
                services["kimi_llm"] = ServiceStatus(
                    status=status_str,
                    latency_ms=round(kimi_latency, 2),
                    message=f"Kimi reachable (HTTP {res.status_code})",
                )
        except Exception as e:
            logger.warning("Health probe Kimi unreachable: %s", e)
            services["kimi_llm"] = ServiceStatus(
                status="degraded",
                message=f"Kimi probe unreachable: {str(e)[:80]}",
            )
    else:
        services["kimi_llm"] = ServiceStatus(
            status="degraded",
            message="Kimi API key unconfigured (set KIMI_API_KEY in .env)",
        )

    # 4. Check Background Workers
    worker_task = getattr(request.app.state, "worker_task", None) if hasattr(request, "app") and hasattr(request.app, "state") else None
    if worker_task and not worker_task.done():
        services["ml_worker"] = ServiceStatus(
            status="healthy",
            message="Background ML scheduler active (anomaly: 60s, forecast: 5m, retention: 1h)",
        )
    elif hasattr(request, "app") and getattr(request.app.state, "worker_started_successfully", False) and worker_task and worker_task.done():
        exc = worker_task.exception() if not worker_task.cancelled() else "cancelled"
        services["ml_worker"] = ServiceStatus(
            status="unhealthy",
            message=f"MLWorker background task crashed: {exc}",
        )
    else:
        services["ml_worker"] = ServiceStatus(
            status="degraded",
            message="MLWorker not running or idle",
        )

    ingestion_task = getattr(request.app.state, "ingestion_task", None) if hasattr(request, "app") and hasattr(request.app, "state") else None
    if ingestion_task and not ingestion_task.done():
        services["ingestion_worker"] = ServiceStatus(
            status="healthy",
            message="Telemetry stream batch processor active",
        )
    else:
        services["ingestion_worker"] = ServiceStatus(
            status="degraded",
            message="Telemetry ingestion worker not running or idle",
        )

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        services=services,
    )

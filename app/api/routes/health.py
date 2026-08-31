import time
import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.db.session import get_db
from app.services.redis_service import redis_service
from app.api.schemas.health import HealthResponse, ServiceStatus

router = APIRouter(prefix="/health", tags=["Observability"])


@router.get("", response_model=HealthResponse)
async def check_health(db: AsyncSession = Depends(get_db)):
    """
    Comprehensive diagnostic health probe checking:
    1. PostgreSQL connection & read latency
    2. Redis connection & ping latency
    3. NVIDIA NIM upstream accessibility
    """
    services = {}
    overall_healthy = True

    # 1. Check PostgreSQL
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
        services["postgresql"] = ServiceStatus(
            status="unhealthy",
            message=f"PostgreSQL connection failed: {str(e)[:100]}",
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
            services["redis"] = ServiceStatus(
                status="unhealthy",
                message=f"Redis ping failed: {str(e)[:100]}",
            )
    else:
        services["redis"] = ServiceStatus(
            status="degraded",
            message="Redis client not connected (running in in-memory fallback)",
        )

    # 3. Check NVIDIA NIM connectivity
    nim_key = settings.NIM_API_KEY_1 or settings.NIM_API_KEY_2 or settings.NIM_API_KEY_3
    if nim_key:
        nim_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{settings.NIM_ENDPOINT_1.rstrip('/')}/models", headers={"Authorization": f"Bearer {nim_key}"})
                nim_latency = (time.perf_counter() - nim_start) * 1000
                services["nvidia_nim"] = ServiceStatus(
                    status="healthy" if res.status_code < 500 else "degraded",
                    latency_ms=round(nim_latency, 2),
                    message=f"NIM reachable (HTTP {res.status_code})",
                )
        except Exception as e:
            services["nvidia_nim"] = ServiceStatus(
                status="degraded",
                message=f"NIM probe unreachable: {str(e)[:80]}",
            )
    else:
        services["nvidia_nim"] = ServiceStatus(
            status="healthy",
            message="NIM API keys unconfigured (running in heuristic fallback mode)",
        )

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        services=services,
    )

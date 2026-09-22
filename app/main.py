import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import time
import uuid

from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import (
    setup_logging,
    logger,
    request_id_ctx,
    correlation_id_ctx,
    business_id_ctx,
)
from app.db.session import engine
from app.db.base import Base
from app.services.redis_service import redis_service
from app.workers.ml_jobs import MLWorker
from app.workers.ingestion_worker import IngestionWorker

# Routers
from app.api.routes.health import router as health_router
from app.api.routes.auth import router as auth_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.friday import router as friday_router
from app.api.routes.users import router as users_router
from app.api.routes.logs import router as logs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Starting %s v%s [%s]", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)
    
    # Connect Redis
    try:
        await redis_service.connect()
    except Exception as e:
        logger.error("Failed to initialize Redis connection during startup: %s", e, exc_info=True)

    # Database initialization & migrations
    try:
        from app.db.session import is_db_available
        if await is_db_available(force_check=True):
            if settings.ENVIRONMENT == "development":
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables initialized successfully (development mode)")
            else:
                from app.db.migrate import run_upgrade
                run_upgrade("head")
                logger.info("Alembic database migrations applied successfully to head")

            # Purge legacy synthetic telemetry rows to ensure pristine database state
            try:
                from sqlalchemy import text
                async with engine.begin() as conn:
                    await conn.execute(text("DELETE FROM telemetry_events WHERE payload_metadata->>'synthetic_baseline' = 'true'"))
            except Exception as pe:
                logger.debug("Synthetic baseline cleanup skipped: %s", pe)
        else:
            db_host = settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "localhost:5432"
            logger.info("PostgreSQL offline on %s; operating in decoupled mode", db_host)
    except Exception as e:
        logger.warning("Database migration during startup skipped or failed: %s; operating in decoupled mode", e)

    # -------------------------------------------------------------
    # Embedded Background Workers (MLWorker + IngestionWorker)
    # -------------------------------------------------------------
    ml_worker: Optional[MLWorker] = None
    worker_task: Optional[asyncio.Task] = None
    ingestion_worker: Optional[IngestionWorker] = None
    ingestion_task: Optional[asyncio.Task] = None
    is_shutting_down: bool = False
    worker_started_successfully: bool = False

    def _worker_done_callback(t: asyncio.Task) -> None:
        if is_shutting_down or t.cancelled():
            return

        exc = t.exception()
        if not worker_started_successfully:
            # Startup failure: handled explicitly in lifespan startup check.
            return

        if exc:
            logger.error(
                "MLWorker background task terminated unexpectedly with error: %s",
                exc,
                exc_info=exc,
            )
        else:
            logger.warning("MLWorker background task exited unexpectedly while application is running.")

    async def _run_worker_guarded() -> None:
        nonlocal worker_started_successfully
        # Pre-check Redis connection availability
        if redis_service.redis is None:
            if settings.ENVIRONMENT == "development":
                logger.info("Redis server offline: running embedded MLWorker with in-memory lock coordination in development mode")
            else:
                raise ConnectionError(
                    "Redis server is unreachable or offline; MLWorker requires Redis for distributed locking and coordination"
                )
        else:
            try:
                await redis_service.redis.ping()
            except Exception as e:
                if settings.ENVIRONMENT == "development":
                    logger.info("Redis ping failed: running embedded MLWorker with in-memory lock fallback: %s", e)
                else:
                    raise ConnectionError(f"Redis ping probe failed: {e}") from e

        # Monitor for forced crash injection during mid-run testing if event configured
        if hasattr(app.state, "_inject_worker_crash_event"):
            crash_task = asyncio.create_task(app.state._inject_worker_crash_event.wait())
            worker_run_task = asyncio.create_task(ml_worker.run())
            done, _ = await asyncio.wait(
                [worker_run_task, crash_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if crash_task in done and app.state._inject_worker_crash_event.is_set():
                worker_run_task.cancel()
                exc_to_raise = getattr(
                    app.state,
                    "_injected_worker_exception",
                    RuntimeError("Forced mid-run worker crash"),
                )
                raise exc_to_raise
            else:
                crash_task.cancel()
                await worker_run_task
        else:
            await ml_worker.run()

    try:
        # Initialize crash injection event hook on app.state
        app.state._inject_worker_crash_event = asyncio.Event()
        app.state.inject_worker_crash = lambda exc=None: (
            setattr(app.state, "_injected_worker_exception", exc or RuntimeError("Forced mid-run worker crash")),
            app.state._inject_worker_crash_event.set(),
        )

        ml_worker = MLWorker(embedded=True)
        worker_task = asyncio.create_task(_run_worker_guarded())
        worker_task.add_done_callback(_worker_done_callback)

        app.state.ml_worker = ml_worker
        app.state.worker_task = worker_task

        # Brief yield to catch immediate startup failures (e.g., in production when Redis is required)
        await asyncio.sleep(0.1)
        if worker_task.done():
            exc = worker_task.exception()
            if exc:
                logger.error("MLWorker failed to start: %s", exc, exc_info=exc)
                worker_task = None
            else:
                logger.warning("MLWorker task completed immediately after startup.")
        else:
            worker_started_successfully = True
            app.state.worker_started_successfully = True
            logger.info("MLWorker embedded scheduler started in background task.")

        # Start IngestionWorker background task
        ingestion_worker = IngestionWorker(embedded=True)
        ingestion_task = asyncio.create_task(ingestion_worker.run())
        app.state.ingestion_worker = ingestion_worker
        app.state.ingestion_task = ingestion_task
        logger.info("IngestionWorker stream processor started in background task.")

    except Exception as e:
        logger.error("Workers failed to start: %s", e, exc_info=True)
        ml_worker = None
        worker_task = None

    yield

    # Shutdown
    is_shutting_down = True
    logger.info("Shutting down AI-CTO Backend...")
    if ml_worker:
        logger.info("Shutting down MLWorker scheduler...")
        ml_worker.stop()

    if ingestion_worker:
        logger.info("Shutting down IngestionWorker...")
        ingestion_worker.stop()

    if worker_task:
        if worker_task.done():
            exc = worker_task.exception()
            if exc:
                logger.error(
                    "MLWorker background task was found dead during shutdown with exception: %s",
                    exc,
                    exc_info=exc,
                )
            else:
                logger.info("MLWorker task had already finished prior to shutdown.")
        else:
            try:
                await asyncio.wait_for(worker_task, timeout=5.0)
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("MLWorker did not terminate within timeout.")
            except Exception as e:
                logger.warning("Error waiting for MLWorker shutdown: %s", e)

    if ingestion_task:
        if not ingestion_task.done():
            try:
                await asyncio.wait_for(ingestion_task, timeout=3.0)
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("IngestionWorker did not terminate within timeout.")
            except Exception as e:
                logger.warning("Error waiting for IngestionWorker shutdown: %s", e)

    try:
        await redis_service.disconnect()
        await engine.dispose()
        logger.info("Database and Redis connections closed.")
    except Exception as e:
        logger.error("Error during shutdown cleanup: %s", e, exc_info=True)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-CTO Modular Monolith Backend API (Reliability-Hardened v2)",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration (Strict baseline for standard API endpoints)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ingestion Route Scoped CORS Middleware (Permissive CORS scoped solely to /ingestion/events)
@app.middleware("http")
async def ingestion_scoped_cors_middleware(request: Request, call_next):
    is_ingestion_path = request.url.path in [
        "/api/ingestion/events",
        "/api/v1/ingestion/events",
        "/ingestion/events",
    ]
    origin = request.headers.get("Origin")

    # Only process requests targeting the ingestion events pipeline
    if not is_ingestion_path:
        return await call_next(request)

    # Handle Preflight OPTIONS probe specifically for ingestion
    if request.method == "OPTIONS":
        if origin:
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, X-API-Key, Authorization, X-Correlation-ID, X-Request-ID",
                    "Access-Control-Max-Age": "86400",
                },
            )
        return Response(status_code=204)

    # For actual ingestion request, execute handler
    response = await call_next(request)

    # Dynamic Origin Validation & Header Reflection for Tenant Telemetry
    if origin:
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")
        tenant_id = None
        if api_key:
            from app.api.routes.ingestion import _resolve_api_key_tenant
            tenant_id = await _resolve_api_key_tenant(api_key.strip())

        # Reflect origin if authenticated via valid tenant API key or Bearer token, or on error responses
        if tenant_id or (auth_header and auth_header.startswith("Bearer ")) or response.status_code in [400, 401, 403, 422]:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization, X-Correlation-ID, X-Request-ID"

    return response


# Request ID & Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    correlation_id_ctx.set(corr_id)
    request_id_ctx.set(corr_id)
    business_id_ctx.set("")
    
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        process_time = (time.perf_counter() - start_time) * 1000
        logger.error(
            "%s %s -> 500 in %.2fms (Unhandled Exception: %s)",
            request.method,
            request.url.path,
            process_time,
            exc,
            exc_info=True,
        )
        raise exc

    process_time = (time.perf_counter() - start_time) * 1000

    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Request-ID"] = corr_id
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

    logger.info(
        "%s %s -> %d in %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        process_time,
    )
    return response


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception on %s %s: %s", request.method, request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact AI-CTO support."},
    )


# Mount API Routers
app.include_router(health_router)
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)
app.include_router(ingestion_router, prefix=settings.API_V1_STR)
app.include_router(friday_router, prefix=settings.API_V1_STR)
app.include_router(users_router, prefix=settings.API_V1_STR)
app.include_router(logs_router, prefix=settings.API_V1_STR)

# Also support /api/v1 routes
# TODO: If settings.API_V1_STR is meant to be the canonical prefix going forward,
# deprecate the non-prefixed (/api) routes rather than maintaining both indefinitely.
if settings.API_V1_STR != "/api/v1":
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(dashboard_router, prefix="/api/v1")
    app.include_router(ingestion_router, prefix="/api/v1")
    app.include_router(friday_router, prefix="/api/v1")
    app.include_router(users_router, prefix="/api/v1")
    app.include_router(logs_router, prefix="/api/v1")



@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
    }


# Static File Hosting & Dedicated Tracker Endpoint
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/static/tracker.js")
@app.get("/api/tracker.js")
@app.get("/tracker.js")
async def serve_tracker_script():
    tracker_path = STATIC_DIR / "tracker.js"
    if tracker_path.exists():
        return FileResponse(
            str(tracker_path),
            media_type="application/javascript",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            },
        )
    raise HTTPException(status_code=404, detail="tracker.js not found")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

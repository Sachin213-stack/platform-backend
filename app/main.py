from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

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

# Routers
from app.api.routes.health import router as health_router
from app.api.routes.auth import router as auth_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.friday import router as friday_router


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

    # Create tables automatically in development mode
    if settings.ENVIRONMENT == "development":
        try:
            from app.db.session import is_db_available
            if await is_db_available():
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables initialized successfully")
            else:
                db_host = settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "localhost:5432"
                logger.info("PostgreSQL offline on %s; operating in decoupled development mode", db_host)
        except Exception as e:
            logger.info("PostgreSQL offline (%s); operating in decoupled development mode", e)

    yield

    # Shutdown
    logger.info("Shutting down AI-CTO Backend...")
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

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)
app.include_router(ingestion_router, prefix=settings.API_V1_STR)
app.include_router(friday_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
    }

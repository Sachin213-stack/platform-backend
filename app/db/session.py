import asyncio
import time
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app.core.config import settings
from app.core.logging import logger

# Create Async Engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

_db_available: Optional[bool] = None
_db_last_checked: float = 0.0


async def _probe_db() -> bool:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


async def is_db_available(force_check: bool = False) -> bool:
    """
    Checks if the primary PostgreSQL database is reachable.
    Caches availability status (20s TTL) to avoid repeating connection timeouts.
    Uses 1.0s probe timeout to fail fast when offline.
    """
    global _db_available, _db_last_checked
    now = time.time()
    cache_ttl = 15.0 if _db_available else 20.0
    if not force_check and _db_available is not None and (now - _db_last_checked < cache_ttl):
        return _db_available

    try:
        await asyncio.wait_for(_probe_db(), timeout=1.0)
        _db_available = True
    except Exception:
        _db_available = False

    _db_last_checked = now
    return _db_available


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Base DB session generator.
    Routes are responsible for calling session.commit() explicitly.
    This dependency handles rollback on exception and session cleanup.
    """
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception as e:
        try:
            await session.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            await session.close()
        except Exception:
            pass


async def set_rls_context(session: AsyncSession, business_id: str) -> None:
    """
    Sets the PostgreSQL session variable for Row-Level Security (RLS).
    PostgreSQL policies check: current_setting('app.current_business_id', true) = business_id
    """
    if business_id:
        try:
            await session.execute(
                text("SELECT set_config('app.current_business_id', :business_id, true)"),
                {"business_id": str(business_id)},
            )
        except Exception as e:
            logger.debug("Could not set RLS context (DB offline): %s", e)

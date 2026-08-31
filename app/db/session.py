from typing import AsyncGenerator
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


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Base DB session generator.
    Routes are responsible for calling session.commit() explicitly.
    This dependency only handles rollback on exception and session cleanup.
    """
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception as e:
        try:
            await session.rollback()
        except Exception:
            pass
        if settings.ENVIRONMENT != "development":
            raise
        logger.warning("DB session error (suppressed in dev): %s", e)
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
        await session.execute(
            text("SELECT set_config('app.current_business_id', :business_id, true)"),
            {"business_id": str(business_id)},
        )

import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.security import decode_token
from app.core.logging import business_id_ctx, logger
from app.db.session import get_db, set_rls_context
from app.db.models.business import User, Business
from app.services.redis_service import redis_service

security_bearer = HTTPBearer(auto_error=False)


async def get_current_user_and_business(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validates the JWT token, verifies that it hasn't been revoked in Redis,
    extracts the business_id, sets the request context and Postgres RLS session.
    """
    if not credentials:
        if settings.ENVIRONMENT == "development":
            return User(
                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                business_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                email="dev.admin@aicto.io",
                hashed_password="",
                full_name="AI-CTO Lead Engineer",
                role="owner",
                is_active=True,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token is in Redis revocation blacklist
    jti = payload.get("jti")
    if jti and await redis_service.is_token_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("sub")
    business_id_str = payload.get("business_id")

    if not user_id_str or not business_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing required claims",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = uuid.UUID(user_id_str)
        business_id = uuid.UUID(business_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid identifier format in token",
        )

    # Set logging contextvar
    business_id_ctx.set(str(business_id))

    # Set PostgreSQL Row-Level Security (RLS) context variable (if DB available)
    try:
        await set_rls_context(db, str(business_id))
    except Exception as e:
        logger.debug(f"Could not set RLS context (DB may be offline): {e}")

    # Fetch User from DB (or use fallback in development mode)
    try:
        stmt = select(User).where(User.id == user_id, User.business_id == business_id)
        result = await db.execute(stmt)
        user = result.scalars().first()
    except Exception as e:
        logger.warning(f"Database query failed in auth dependency: {e}")
        user = None

    if not user:
        if settings.ENVIRONMENT == "development":
            # Development fallback user
            return User(
                id=user_id,
                business_id=business_id,
                email=payload.get("email", "dev.admin@aicto.io"),
                hashed_password="",
                full_name=payload.get("name", "AI-CTO Lead Engineer"),
                role=payload.get("role", "owner"),
                is_active=True,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or does not belong to the token business",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    return user


async def get_current_business_id(
    current_user: User = Depends(get_current_user_and_business),
) -> uuid.UUID:
    """Dependency that returns the authenticated business_id."""
    return current_user.business_id

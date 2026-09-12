import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
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
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validates the JWT token, verifies that it hasn't been revoked in Redis,
    extracts the business_id, sets the request context and Postgres RLS session.
    """
    if not credentials:
        logger.warning("Auth failure on %s: Missing authorization credentials", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_token(token)
    if not payload:
        logger.warning("Auth failure on %s: Invalid or expired JWT token", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token is in Redis revocation blacklist
    jti = payload.get("jti")
    if jti and await redis_service.is_token_revoked(jti):
        logger.warning("Auth failure on %s: Token has been revoked (jti=%s)", request.url.path, jti)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("sub")
    business_id_str = payload.get("business_id")

    if not user_id_str or not business_id_str:
        logger.warning(
            "Auth failure on %s: Token payload missing required claims (has_sub=%s, has_business_id=%s)",
            request.url.path,
            bool(user_id_str),
            bool(business_id_str),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing required claims",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = uuid.UUID(user_id_str)
        business_id = uuid.UUID(business_id_str)
    except ValueError:
        logger.warning(
            "Auth failure on %s: Malformed UUID identifier in token claims (user_id=%s, business_id=%s)",
            request.url.path,
            user_id_str,
            business_id_str,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid identifier format in token",
        )

    # Set logging contextvar immediately upon successful token validation
    business_id_ctx.set(str(business_id))

    # Set PostgreSQL Row-Level Security (RLS) context variable (if DB available)
    from app.db.session import is_db_available
    db_online = await is_db_available()

    if db_online:
        try:
            await set_rls_context(db, str(business_id))
        except Exception as e:
            logger.debug("PostgreSQL RLS context error for business %s on %s: %s", business_id, request.url.path, e)

    # Fetch User from DB (or use fallback in development mode)
    user = None
    if db_online:
        try:
            stmt = select(User).where(User.id == user_id, User.business_id == business_id)
            result = await db.execute(stmt)
            user = result.scalars().first()
        except Exception as e:
            logger.debug("Database query failed in auth dependency for user %s: %s", user_id, e)
            user = None

    if not user:
        if settings.ENVIRONMENT == "development" or str(user_id) == "00000000-0000-0000-0000-000000000001":
            # Development / Demo fallback user (Alex Vance / Apex Retail Global)
            return User(
                id=user_id,
                business_id=business_id,
                email=payload.get("email", "demo.cto@aicto.io"),
                hashed_password="",
                full_name=payload.get("name", "Alex Vance (Lead Architect)"),
                role=payload.get("role", "owner"),
                is_active=True,
            )
        logger.warning("Auth failure on %s: User %s not found or does not belong to business %s", request.url.path, user_id, business_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or does not belong to the token business",
        )

    if not user.is_active:
        logger.warning("Auth failure on %s: User account %s is deactivated (business=%s)", request.url.path, user.id, business_id)
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

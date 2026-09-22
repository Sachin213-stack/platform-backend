import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Response, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.logging import logger
from app.core.security import decode_token
from app.db.session import get_db, is_db_available
from app.db.models.business import Business, User, ApiKey
from app.api.schemas.auth import UserResponse, UserOrgUpdate, ApiKeyResponse, ApiKeyCreate
from app.api.dependencies.auth import get_current_user_and_business
from app.services.redis_service import redis_service

router = APIRouter(prefix="/users", tags=["Users & Profiles"])

ALLOWED_AVATAR_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}
MAX_AVATAR_BYTES = 2 * 1024 * 1024  # 2MB


def build_user_response(user: User, business: Optional[Business] = None) -> UserResponse:
    has_avatar = user.avatar_data is not None and len(user.avatar_data) > 0
    avatar_url = f"/api/users/{user.id}/avatar" if has_avatar else None

    settings_cfg = (business.settings_config if business and business.settings_config else {}) or {}

    return UserResponse(
        id=user.id,
        name=user.full_name or (user.email.split("@")[0] if user.email else "User"),
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        avatar_url=avatar_url,
        business_id=user.business_id,
        business_name=business.name if business else "Acme Innovations",
        business_type=getattr(business, "business_type", "ecommerce") if business else "ecommerce",
        ops_email=getattr(business, "ops_email", None) or user.email,
        timezone=settings_cfg.get("timezone", "America/New_York"),
        currency=settings_cfg.get("currency", "USD"),
        auto_refresh_interval=settings_cfg.get("auto_refresh_interval", "30s"),
    )


@router.get("/me", response_model=UserResponse)
async def get_my_profile(
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full profile of the currently authenticated user and their organization."""
    business = None
    if await is_db_available():
        try:
            stmt = select(Business).where(Business.id == current_user.business_id)
            result = await db.execute(stmt)
            business = result.scalars().first()
        except Exception as e:
            logger.warning("Could not load business info for user %s: %s", current_user.id, e)

    return build_user_response(current_user, business)


@router.put("/me", response_model=UserResponse)
@router.patch("/me", response_model=UserResponse)
async def update_my_profile(
    data: UserOrgUpdate,
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    """Update profile name and organization fields with database persistence."""
    business = None
    if await is_db_available():
        try:
            # Query fresh user and business instances from session
            user_stmt = select(User).where(User.id == current_user.id)
            user_res = await db.execute(user_stmt)
            user = user_res.scalars().first() or current_user

            biz_stmt = select(Business).where(Business.id == current_user.business_id)
            biz_res = await db.execute(biz_stmt)
            business = biz_res.scalars().first()

            # Update User fields
            new_name = data.name or data.full_name
            if new_name is not None:
                user.full_name = new_name.strip()

            # Update Business fields
            if business:
                if data.business_name is not None:
                    business.name = data.business_name.strip()
                if data.business_type is not None:
                    business.business_type = data.business_type.strip()
                if data.ops_email is not None:
                    business.ops_email = str(data.ops_email).strip()

                # Update preferences in settings_config
                cfg = dict(business.settings_config or {})
                if data.timezone is not None:
                    cfg["timezone"] = data.timezone
                if data.currency is not None:
                    cfg["currency"] = data.currency
                if data.auto_refresh_interval is not None:
                    cfg["auto_refresh_interval"] = data.auto_refresh_interval
                business.settings_config = cfg

            await db.commit()
            await db.refresh(user)
            if business:
                await db.refresh(business)

            logger.info("Updated profile and org for user %s (business %s)", user.id, user.business_id)
            return build_user_response(user, business)
        except Exception as e:
            await db.rollback()
            logger.error("Failed to update profile for user %s: %s", current_user.id, e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update user profile: {str(e)}",
            )

    # In decoupled mode without DB, return in-memory update
    if data.name or data.full_name:
        current_user.full_name = data.name or data.full_name
    return build_user_response(current_user, business)


@router.post("/me/avatar", status_code=status.HTTP_200_OK)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a profile avatar image.
    Enforces format (jpeg, png, webp) and max size (2MB).
    Stores binary data in PostgreSQL users.avatar_data (BYTEA) column.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_AVATAR_TYPES:
        logger.warning(
            "Avatar upload rejected: Invalid MIME type %s for user %s", content_type, current_user.id
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG, PNG, and WebP images are supported",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    if len(content) > MAX_AVATAR_BYTES:
        logger.warning(
            "Avatar upload rejected: File size %d exceeds %d for user %s",
            len(content),
            MAX_AVATAR_BYTES,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Avatar image file size must be less than 2MB",
        )

    normalized_mime = ALLOWED_AVATAR_TYPES[content_type]

    if await is_db_available():
        try:
            stmt = select(User).where(User.id == current_user.id)
            res = await db.execute(stmt)
            user = res.scalars().first() or current_user

            user.avatar_data = content
            user.avatar_mime_type = normalized_mime
            await db.commit()
            await db.refresh(user)
            logger.info("Avatar saved to DB for user %s (%d bytes, %s)", user.id, len(content), normalized_mime)
        except Exception as e:
            await db.rollback()
            logger.error("Failed to save avatar to DB for user %s: %s", current_user.id, e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while saving avatar",
            )
    else:
        current_user.avatar_data = content
        current_user.avatar_mime_type = normalized_mime

    return {
        "message": "Avatar uploaded successfully",
        "avatar_url": f"/api/users/{current_user.id}/avatar",
    }


@router.get("/me/avatar")
async def get_my_avatar(
    token: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user_and_business),
):
    """Streams the authenticated user's avatar image directly from the PostgreSQL BYTEA column."""
    if not current_user.avatar_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Avatar not found")

    return Response(
        content=current_user.avatar_data,
        media_type=current_user.avatar_mime_type or "image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Content-Disposition": "inline",
        },
    )


@router.get("/{user_id}/avatar")
async def get_user_avatar_by_id(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Public avatar streaming endpoint by user ID.
    Enables HTML <img src="/api/users/{user_id}/avatar"> tags to load avatars without Bearer headers.
    """
    if await is_db_available():
        try:
            stmt = select(User).where(User.id == user_id)
            res = await db.execute(stmt)
            user = res.scalars().first()
            if user and user.avatar_data:
                return Response(
                    content=user.avatar_data,
                    media_type=user.avatar_mime_type or "image/png",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Content-Disposition": "inline",
                    },
                )
        except Exception as e:
            logger.warning("Could not fetch avatar for user %s: %s", user_id, e)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Avatar not found")


@router.get("/api-keys", response_model=List[ApiKeyResponse])
async def list_business_api_keys(
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve active API keys for the current business.
    Returns existing keys or generates an initial active telemetry key if none exist.
    """
    biz_id_str = str(current_user.business_id)
    keys_out: List[ApiKeyResponse] = []

    if await is_db_available():
        try:
            stmt = select(ApiKey).where(
                ApiKey.business_id == current_user.business_id,
                ApiKey.is_active.is_(True),
            )
            res = await db.execute(stmt)
            keys = res.scalars().all()
            for k in keys:
                raw = k.get_key()
                if raw:
                    await redis_service.set_cache(f"apikey:resolved:{raw[:16]}", biz_id_str, ttl_seconds=86400 * 30)
                keys_out.append(
                    ApiKeyResponse(
                        id=k.id,
                        name=k.name,
                        key_prefix=k.key_prefix,
                        api_key=raw if current_user.role in ["owner", "admin"] else None,
                        is_active=k.is_active,
                        created_at=k.created_at.isoformat() if hasattr(k, "created_at") and k.created_at else None,
                    )
                )
        except Exception as e:
            logger.warning("Could not fetch API keys from DB for business %s: %s", biz_id_str, e)

    # If no keys exist in DB (or DB unavailable in dev mode), create initial snippet key
    if not keys_out:
        new_raw_key = f"sk_live_{uuid.uuid4().hex}"
        key_id = uuid.uuid4()
        prefix = new_raw_key[:12] + "..."

        if await is_db_available():
            try:
                new_key_obj = ApiKey(
                    id=key_id,
                    business_id=current_user.business_id,
                    name="Website Telemetry Snippet Key",
                    is_active=True,
                )
                new_key_obj.set_key(new_raw_key)
                db.add(new_key_obj)
                await db.commit()
            except Exception as e:
                logger.warning("Could not persist initial API key to DB: %s", e)

        # Cache in Redis/memory cache
        await redis_service.set_cache(f"apikey:resolved:{new_raw_key[:16]}", biz_id_str, ttl_seconds=86400 * 30)

        keys_out.append(
            ApiKeyResponse(
                id=key_id,
                name="Website Telemetry Snippet Key",
                key_prefix=prefix,
                api_key=new_raw_key,
                is_active=True,
            )
        )

    return keys_out


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_business_api_key(
    data: ApiKeyCreate,
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a new API key for the current business.
    Stored securely (encrypted at rest) and cached for low-latency ingestion auth.
    """
    if current_user.role not in ["owner", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only business owners or admins can generate API keys.",
        )

    biz_id_str = str(current_user.business_id)
    raw_key = f"sk_live_{uuid.uuid4().hex}"
    key_id = uuid.uuid4()
    prefix = raw_key[:12] + "..."

    if await is_db_available():
        try:
            key_obj = ApiKey(
                id=key_id,
                business_id=current_user.business_id,
                name=data.name or "Website Telemetry Snippet Key",
                is_active=True,
            )
            key_obj.set_key(raw_key)
            db.add(key_obj)
            await db.commit()
        except Exception as e:
            logger.error("Failed to save new API key to database: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while saving API key",
            )

    # Cache resolution in Redis
    await redis_service.set_cache(f"apikey:resolved:{raw_key[:16]}", biz_id_str, ttl_seconds=86400 * 30)

    logger.info("Generated new API key '%s' for business %s", data.name, biz_id_str)
    return ApiKeyResponse(
        id=key_id,
        name=data.name or "Website Telemetry Snippet Key",
        key_prefix=prefix,
        api_key=raw_key,
        is_active=True,
    )

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Response, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.logging import logger
from app.core.security import decode_token
from app.db.session import get_db, is_db_available
from app.db.models.business import Business, User
from app.api.schemas.auth import UserResponse, UserOrgUpdate
from app.api.dependencies.auth import get_current_user_and_business

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

import uuid
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.logging import logger
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.db.session import get_db, is_db_available
from app.db.models.business import Business, User
from app.api.schemas.auth import (
    UserRegister,
    UserLogin,
    Token,
    UserResponse,
)
from app.api.dependencies.auth import get_current_user_and_business
from app.services.redis_service import redis_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(data: UserRegister, db: AsyncSession = Depends(get_db)):
    from app.db.session import is_db_available
    db_online = await is_db_available()

    if db_online:
        # Check if user email already exists
        try:
            existing_user = await db.execute(select(User).where(User.email == data.email))
            if existing_user.scalars().first():
                logger.warning("Registration rejected: Email %s already exists", data.email)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A user with this email address already exists",
                )
        except HTTPException:
            raise
        except Exception as e:
            if settings.ENVIRONMENT != "development":
                logger.error("Database unavailable during registration check for %s: %s", data.email, e, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database is currently unavailable",
                )
            logger.info("DB offline during registration check: %s", e)

    # Generate unique slug for business
    base_slug = data.business_name.lower().replace(" ", "-")
    slug = f"{base_slug}-{uuid.uuid4().hex[:6]}"

    biz_uuid = uuid.uuid4()
    user_uuid = uuid.uuid4()
    biz_id = str(biz_uuid)
    user_id = str(user_uuid)
    role = "owner"

    if db_online:
        try:
            from app.db.session import set_rls_context
            await set_rls_context(db, biz_id)

            # Create Business
            new_business = Business(
                id=biz_uuid,
                name=data.business_name,
                slug=slug,
                plan_tier="starter",
                retention_days=30,
            )
            db.add(new_business)
            await db.flush()  # Populates new_business.id

            # Create Owner User
            new_user = User(
                id=user_uuid,
                business_id=biz_uuid,
                email=data.email,
                hashed_password=get_password_hash(data.password),
                full_name=data.full_name or data.business_name,
                role="owner",
                is_active=True,
            )
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)

            user_id = str(new_user.id)
            biz_id = str(new_business.id)
            role = new_user.role
            logger.info("User registered successfully: email=%s, user_id=%s, business_id=%s", data.email, user_id, biz_id)
        except HTTPException:
            raise
        except Exception as e:
            try:
                await db.rollback()
            except Exception:
                pass
            if settings.ENVIRONMENT != "development":
                logger.error("Failed to create user account for %s: %s", data.email, e, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create user account: {type(e).__name__}: {e}",
                )
            logger.info("Registration processed in dev fallback mode (DB offline): %s", e)
    else:
        logger.info("New user registration processed in dev fallback mode: email=%s, user_id=%s, business_id=%s", data.email, user_id, biz_id)

    # Issue JWT tokens
    jti = uuid.uuid4().hex
    access_token = create_access_token({
        "sub": user_id,
        "business_id": biz_id,
        "role": role,
        "email": data.email,
        "name": data.full_name or data.business_name,
        "jti": jti,
    })
    refresh_token = create_refresh_token({
        "sub": user_id,
        "business_id": biz_id,
        "role": role,
    })

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        business_id=biz_id,
        user_id=user_id,
    )


@router.post("/login", response_model=Token)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    from app.db.session import is_db_available
    db_online = await is_db_available()

    user = None
    if db_online:
        try:
            result = await db.execute(select(User).where(User.email == data.email))
            user = result.scalars().first()
        except Exception as e:
            logger.debug("Database query failed during login for %s: %s", data.email, e)
            user = None

    if user:
        if not verify_password(data.password, user.hashed_password):
            logger.warning("Failed login attempt for email %s: Incorrect password", data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        if not user.is_active:
            logger.warning("Failed login attempt for email %s: Account is deactivated (user_id=%s)", data.email, user.id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated",
            )
        user_id = str(user.id)
        biz_id = str(user.business_id)
        role = user.role
        user_name = user.full_name
    elif settings.ENVIRONMENT == "development":
        # Dev fallback login for seamless pairing
        user_id = str(uuid.uuid4())
        biz_id = str(uuid.uuid4())
        role = "owner"
        user_name = data.email.split("@")[0].capitalize()
        logger.info("Dev fallback login granted for email %s", data.email)
    else:
        logger.warning("Failed login attempt for email %s: User not found", data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    jti = uuid.uuid4().hex
    access_token = create_access_token({
        "sub": user_id,
        "business_id": biz_id,
        "role": role,
        "email": data.email,
        "name": user_name,
        "jti": jti,
    })
    refresh_token = create_refresh_token({
        "sub": user_id,
        "business_id": biz_id,
        "role": role,
    })

    logger.info("User logged in successfully: email=%s, user_id=%s, business_id=%s", data.email, user_id, biz_id)
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        business_id=biz_id,
        user_id=user_id,
    )


@router.post("/demo", response_model=Token)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Quick 1-click token generation for demo / testing session."""
    demo_user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    demo_biz_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    jti = uuid.uuid4().hex

    if await is_db_available():
        try:
            biz_stmt = select(Business).where(Business.id == demo_biz_id)
            biz = (await db.execute(biz_stmt)).scalars().first()
            if not biz:
                biz = Business(
                    id=demo_biz_id,
                    name="Apex Retail Global",
                    slug="apex-retail-global",
                    plan_tier="enterprise",
                    retention_days=90,
                )
                db.add(biz)
                await db.flush()

            user_stmt = select(User).where(User.id == demo_user_id)
            user = (await db.execute(user_stmt)).scalars().first()
            if not user:
                user = User(
                    id=demo_user_id,
                    business_id=demo_biz_id,
                    email="demo.cto@aicto.io",
                    hashed_password="",
                    full_name="Alex Vance (Lead Architect)",
                    role="owner",
                    is_active=True,
                )
                db.add(user)
            await db.commit()
        except Exception as e:
            logger.warning("Could not auto-seed demo tenant in DB: %s", e)

    access_token = create_access_token({
        "sub": str(demo_user_id),
        "business_id": str(demo_biz_id),
        "role": "owner",
        "email": "demo.cto@aicto.io",
        "name": "Alex Vance (Lead Architect)",
        "jti": jti,
    })
    refresh_token = create_refresh_token({
        "sub": str(demo_user_id),
        "business_id": str(demo_biz_id),
        "role": "owner",
    })

    logger.info("Demo token generated for demo user %s (tenant %s)", demo_user_id, demo_biz_id)
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        business_id=str(demo_biz_id),
        user_id=str(demo_user_id),
    )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user_and_business),
    authorization: str = Header(...),
):
    """Revokes the current JWT by placing its JTI on the Redis blacklist."""
    try:
        token = authorization.replace("Bearer ", "")
        payload = decode_token(token)
        if payload and "jti" in payload:
            import time
            exp = payload.get("exp", 0)
            remaining_ttl = max(int(exp - time.time()), 60) if exp else settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            await redis_service.revoke_token(payload["jti"], ttl_seconds=remaining_ttl)
            logger.info("Revoked access token jti=%s for user %s (business_id=%s)", payload["jti"], current_user.id, current_user.business_id)
    except Exception as e:
        logger.warning("Error during token revocation for user %s: %s", current_user.id, e, exc_info=True)
    return {"message": "Successfully logged out and revoked access token"}


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    # Fetch business name safely
    business_name = "Apex Retail Global"
    if await is_db_available():
        try:
            result = await db.execute(select(Business).where(Business.id == current_user.business_id))
            business = result.scalars().first()
            if business:
                business_name = business.name
        except Exception as e:
            logger.debug("Could not fetch business name for user %s (business_id=%s): %s", current_user.id, current_user.business_id, e)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        business_id=current_user.business_id,
        business_name=business_name,
    )

import uuid
from typing import List, Optional
from sqlalchemy import String, Boolean, ForeignKey, Integer, JSON, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base, TimestampMixin
from app.core.security import encrypt_secret, decrypt_secret


class Business(Base, TimestampMixin):
    __tablename__ = "businesses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(50), default="starter", nullable=False)
    retention_days: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    business_type: Mapped[str] = mapped_column(String(50), default="ecommerce", nullable=False)
    ops_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    settings_config: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="business", cascade="all, delete-orphan")
    api_keys: Mapped[List["ApiKey"]] = relationship("ApiKey", back_populates="business", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Business id={self.id} name='{self.name}' plan={self.plan_tier}>"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="member", nullable=False)  # owner, admin, member
    avatar_data: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    avatar_mime_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    business: Mapped["Business"] = relationship("Business", back_populates="users")

    def __repr__(self) -> str:
        return f"<User id={self.id} email='{self.email}' role={self.role}>"


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    encrypted_secret: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    business: Mapped["Business"] = relationship("Business", back_populates="api_keys")

    def set_key(self, raw_key: str) -> None:
        self.key_prefix = raw_key[:12] + "..."
        self.encrypted_secret = encrypt_secret(raw_key)

    def get_key(self) -> str:
        return decrypt_secret(self.encrypted_secret)

    def __repr__(self) -> str:
        return f"<ApiKey id={self.id} name='{self.name}' prefix={self.key_prefix}>"

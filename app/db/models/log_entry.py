import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Text, JSON, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class LogEntry(Base):
    __tablename__ = "log_entries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=False
    )
    log_type: Mapped[str] = mapped_column(
        String(20), index=True, nullable=False, default="application"
    )  # application, container, kubernetes, other
    source: Mapped[Optional[str]] = mapped_column(
        String(512), index=True, nullable=True
    )  # service name, pod ID, or file stream
    format: Mapped[str] = mapped_column(
        String(20), nullable=False, default="text"
    )  # json, text, structured
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_fields: Mapped[Optional[dict]] = mapped_column(JSON, default=dict, nullable=True)
    level: Mapped[Optional[str]] = mapped_column(
        String(20), index=True, nullable=True, default="info"
    )  # info, warn, error, debug
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_log_entries_biz_time", "business_id", "timestamp"),
        Index("idx_log_entries_biz_level_time", "business_id", "level", "timestamp"),
        Index("idx_log_entries_biz_type_time", "business_id", "log_type", "timestamp"),
        Index("idx_log_entries_biz_source_time", "business_id", "source", "timestamp"),
    )

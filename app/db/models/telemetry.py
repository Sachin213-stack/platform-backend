import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Float, Integer, JSON, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=False
    )
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, index=True, nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(100), index=True, nullable=False)  # request, order, error, etc.
    
    # Core performance metrics
    response_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    orders_count: Mapped[Optional[int]] = mapped_column(Integer, default=0, nullable=True)
    revenue_amount: Mapped[Optional[float]] = mapped_column(Float, default=0.0, nullable=True)
    cpu_usage_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_usage_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    queue_depth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Detailed metadata
    endpoint: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    payload_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    __table_args__ = (
        Index("idx_telemetry_biz_time", "business_id", "timestamp"),
        Index("idx_telemetry_biz_type_time", "business_id", "event_type", "timestamp"),
    )

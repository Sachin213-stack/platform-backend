import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Float, JSON, DateTime, Index, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base, TimestampMixin


class Anomaly(Base, TimestampMixin):
    __tablename__ = "anomalies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)  # response_time, error_rate, checkout_failures
    severity: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)  # low, medium, high, critical
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    actual_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    __table_args__ = (
        Index("idx_anomalies_biz_detected", "business_id", "detected_at"),
    )


class Forecast(Base, TimestampMixin):
    __tablename__ = "forecasts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    business_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), index=True, nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)  # traffic_24h, revenue_30d, crash_risk
    forecast_horizon: Mapped[str] = mapped_column(String(50), default="24h", nullable=False)  # 24h, 7d, 30d
    crash_risk_pct: Mapped[Optional[float]] = mapped_column(Float, default=0.0, nullable=True)
    forecast_curve: Mapped[dict] = mapped_column(JSON, default=dict)  # timestamps, yhat, yhat_lower, yhat_upper
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    __table_args__ = (
        Index("idx_forecasts_biz_generated", "business_id", "generated_at"),
    )

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TelemetryEventCreate(BaseModel):
    business_id: Optional[str] = Field(None, description="Optional tenant Business ID for beacon scripts")
    idempotency_key: Optional[str] = Field(None, description="Client generated UUID to prevent duplicate metrics")
    event_type: str = Field(..., description="e.g., request, order, error, performance, page_view")
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    orders_count: Optional[int] = 0
    revenue_amount: Optional[float] = 0.0
    cpu_usage_pct: Optional[float] = None
    memory_usage_pct: Optional[float] = None
    queue_depth: Optional[int] = None
    endpoint: Optional[str] = None
    payload_metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class IngestionResponse(BaseModel):
    status: str
    message: str
    event_id: Optional[str] = None
    duplicate: bool = False

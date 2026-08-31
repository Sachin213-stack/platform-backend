from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class KPISummary(BaseModel):
    response_time_ms: float
    response_time_delta_pct: float
    error_rate_pct: float
    error_rate_delta_pct: float
    orders_per_min: float
    orders_delta_pct: float
    checkout_failure_pct: float
    checkout_failure_delta_pct: float


class CapacityMetrics(BaseModel):
    cpu_pct: float
    memory_pct: float
    queue_depth: int


class AnomalyItem(BaseModel):
    id: str
    metric_name: str
    severity: str
    expected_value: float
    actual_value: float
    description: str
    detected_at: datetime


class DashboardMetricsResponse(BaseModel):
    kpis: KPISummary
    capacity: CapacityMetrics
    recent_anomalies: List[AnomalyItem]
    cache_hit: bool = False

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


class TelemetryEventItem(BaseModel):
    id: str
    event_type: str
    endpoint: Optional[str] = None
    response_time_ms: float = 0.0
    status_code: int = 200
    timestamp: datetime
    message: Optional[str] = None
    level: str = "info"


class TimeseriesPoint(BaseModel):
    timestamp: str
    traffic: float
    revenue: float
    response_time_ms: float = 0.0
    error_rate: float = 0.0


class DashboardMetricsResponse(BaseModel):
    kpis: KPISummary
    capacity: CapacityMetrics
    recent_anomalies: List[AnomalyItem]
    recent_telemetry_events: List[TelemetryEventItem] = []
    timeseries: List[TimeseriesPoint] = []
    tier_metrics: Optional[Dict[str, Any]] = None
    has_live_data: bool = False
    total_events_count: int = 0
    cache_hit: bool = False


class AnalyticsSummaryResponse(BaseModel):
    has_live_data: bool = False
    forecast_curve: Dict[str, Any] = {}
    crash_risk_pct: float = 4.2
    resource_runway_days: int = 28
    growth_rate_pct: float = 4.8
    exhaustion_date: str = "In ~28 days"
    bottleneck: str = "Redis Session Store"
    recommended_action: str = "Scale pod replicas & provision memory buffer"
    model_metrics: Dict[str, Any] = {}
    anomalies: List[Dict[str, Any]] = []
    correlation_data: Dict[str, Any] = {}


class AuditLogEntry(BaseModel):
    id: str
    timestamp: str
    actor: str
    action: str
    impact: str
    confidence: str = "99.0%"
    status: str = "Applied"
    service: Optional[str] = None
    action_type: Optional[str] = None


class AuditLogResponse(BaseModel):
    total: int
    entries: List[AuditLogEntry] = []

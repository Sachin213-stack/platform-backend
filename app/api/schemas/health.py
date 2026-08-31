from typing import Dict, Optional
from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    services: Dict[str, ServiceStatus]

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class LogEntryItem(BaseModel):
    id: str
    business_id: str
    log_type: str = "application"
    source: Optional[str] = "app-service"
    format: str = "text"
    content: str
    parsed_fields: Optional[Dict[str, Any]] = None
    level: Optional[str] = "info"
    timestamp: datetime
    ingested_at: Optional[datetime] = None


class LogQueryResponse(BaseModel):
    total: int
    entries: List[LogEntryItem]
    limit: int
    offset: int


class LogSourcesResponse(BaseModel):
    sources: List[str]


class LogAroundAnomalyResponse(BaseModel):
    anomaly_id: str
    anomaly_title: Optional[str] = None
    metric_name: Optional[str] = None
    severity: Optional[str] = None
    detected_at: Optional[datetime] = None
    window_start: datetime
    window_end: datetime
    entries: List[LogEntryItem]
    root_cause_entry_ids: List[str] = Field(default_factory=list)


class LogIngestionItem(BaseModel):
    timestamp: Optional[datetime] = None
    level: Optional[str] = "info"
    source: Optional[str] = "application"
    log_type: Optional[str] = "application"
    format: Optional[str] = "text"
    content: str
    parsed_fields: Optional[Dict[str, Any]] = None


class LogIngestionRequest(BaseModel):
    logs: List[LogIngestionItem]


class LogIngestionResponse(BaseModel):
    status: str = "success"
    ingested_count: int

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant or system or tool")
    content: str = Field(..., description="Message text")


class FridayChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Existing conversation UUID or identifier")
    message: str = Field(..., description="User prompt, engineering query, or voice directive")
    mode: Optional[str] = Field("chat", description="chat or voice")
    model: Optional[str] = Field(None, description="Requested Kimi model tier (e.g. kimi-k3, kimi-k2.6, moonshot-v1-128k)")
    context_hints: Optional[Dict[str, Any]] = Field(None, description="Dynamic widget or incident context (e.g. from Analytics or Dashboard)")
    stream: Optional[bool] = Field(False, description="Whether to request streaming response")


class FridayChatResponse(BaseModel):
    conversation_id: str
    response: str
    model_used: str
    tokens_used: Optional[int] = None
    cached: bool = False
    suggested_actions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Structured actions proposed by Kimi requiring operator confirmation")
    grounding_sources: Optional[List[str]] = Field(default_factory=list, description="Telemetry sources consulted (e.g. 'anomalies_table', 'p99_latency_stream')")


class FridayActionExecutionRequest(BaseModel):
    action_type: str = Field(..., description="scale_service | purge_cdn_cache | throttle_rate_limits | restart_pod_pool | adjust_alert_threshold")
    service: str = Field(..., description="Target service identifier, e.g. checkout-v2")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution parameters (e.g. replicas: 8)")
    conversation_id: Optional[str] = Field(None, description="Conversation session to log the action under")


class FridayActionExecutionResponse(BaseModel):
    action_id: str
    success: bool
    message: str
    executed_at: str
    details: Optional[Dict[str, Any]] = None

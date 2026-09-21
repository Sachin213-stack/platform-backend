import re
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, constr


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant or system or tool")
    content: str = Field(..., description="Message text")


class FridayChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, max_length=128, description="Existing conversation UUID or identifier")
    message: str = Field(..., min_length=1, max_length=4000, description="User prompt, engineering query, or voice directive")
    mode: Optional[str] = Field("chat", description="chat or voice")
    model: Optional[str] = Field(None, max_length=64, description="Requested Kimi model tier (e.g. kimi-k3, kimi-k2.6, moonshot-v1-128k)")
    context_hints: Optional[Dict[str, Any]] = Field(None, description="Dynamic widget or incident context (e.g. from Analytics or Dashboard)")
    stream: Optional[bool] = Field(False, description="Whether to request streaming response")

    @field_validator("context_hints")
    @classmethod
    def validate_context_hints(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is not None:
            if len(v) > 50:
                raise ValueError("context_hints may not contain more than 50 keys.")
            import json
            try:
                serialized = json.dumps(v)
                if len(serialized) > 16384:
                    raise ValueError("context_hints payload size exceeds 16KB limit.")
            except (TypeError, OverflowError):
                raise ValueError("context_hints must be JSON-serializable.")
        return v


class FridayChatResponse(BaseModel):
    conversation_id: str
    response: str
    model_used: str
    tokens_used: Optional[int] = None
    cached: bool = False
    suggested_actions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Structured actions proposed by Kimi requiring operator confirmation")
    grounding_sources: Optional[List[str]] = Field(default_factory=list, description="Telemetry sources consulted (e.g. 'anomalies_table', 'p99_latency_stream')")

class FridayActionExecutionRequest(BaseModel):
    action_type: Literal[
        "scale_service",
        "purge_cdn_cache",
        "throttle_rate_limits",
        "restart_pod_pool",
        "adjust_alert_threshold",
    ] = Field(..., description="Allowed mitigation action types")
    service: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Target service identifier, e.g. checkout-v2 (letters, digits, underscore, hyphen only)",
    )
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution parameters (e.g. replicas: 8)")
    conversation_id: Optional[str] = Field(None, description="Conversation session to log the action under")

    @field_validator("params")
    @classmethod
    def validate_action_params(cls, v: Optional[Dict[str, Any]], info) -> Dict[str, Any]:
        params = v or {}
        action_type = info.data.get("action_type")

        if action_type == "scale_service":
            replicas = params.get("replicas", 6)
            if not isinstance(replicas, int) or isinstance(replicas, bool):
                raise ValueError("Parameter 'replicas' must be an integer.")
            if replicas < 1 or replicas > 30:
                raise ValueError("Parameter 'replicas' must be between 1 and 30.")

        elif action_type == "throttle_rate_limits":
            limit = params.get("limit_per_sec", 250)
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise ValueError("Parameter 'limit_per_sec' must be an integer.")
            if limit < 10 or limit > 50000:
                raise ValueError("Parameter 'limit_per_sec' must be between 10 and 50000 req/sec.")

        return params


class FridayActionExecutionResponse(BaseModel):
    action_id: str
    success: bool
    message: str
    executed_at: str
    details: Optional[Dict[str, Any]] = None


class FridayVoiceSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500, description="Text to synthesize into neural speech")
    voice: Optional[str] = Field("en-US-AriaNeural", max_length=64, description="Voice profile ID or Edge-TTS voice name")
    rate: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Playback rate multiplier (0.5 to 2.0)")
    pitch: Optional[str] = Field("+0Hz", max_length=16, description="Pitch adjustment")


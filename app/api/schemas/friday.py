from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant or system")
    content: str = Field(..., description="Message text")


class FridayChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Existing conversation UUID")
    message: str = Field(..., description="User prompt or query")
    mode: Optional[str] = Field("chat", description="chat or voice")
    context_hints: Optional[Dict[str, Any]] = None


class FridayChatResponse(BaseModel):
    conversation_id: str
    response: str
    model_used: str
    tokens_used: Optional[int] = None
    cached: bool = False

import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.dependencies.auth import get_current_user_and_business
from app.db.session import get_db
from app.db.models.business import User
from app.db.models.alerts import Conversation
from app.api.schemas.friday import FridayChatRequest, FridayChatResponse
from app.services.redis_service import redis_service
from app.services.llm_adapter import llm_adapter
from app.core.logging import logger

router = APIRouter(prefix="/friday", tags=["FRIDAY AI CTO"])

FRIDAY_SYSTEM_PROMPT = """You are FRIDAY, an intelligent AI Assistant & CTO.
You provide clear, accurate, and helpful answers to any general question (technology, programming, concepts, general knowledge, or conversational queries) in natural, generalized language.
If the user specifically asks about system vitals, server performance, or architecture monitoring, use the live system vitals context provided."""


@router.post("/chat", response_model=FridayChatResponse)
async def chat_with_friday(
    request: FridayChatRequest,
    current_user: User = Depends(get_current_user_and_business),
    db: AsyncSession = Depends(get_db),
):
    """
    FRIDAY AI-CTO Conversational Hot Path:
    1. Retrieve conversational session history from Redis (or DB).
    2. Append current user message.
    3. Invoke LLMAdapter with live telemetry context, resilient Multi-NIM fallback & circuit-breaker.
    4. Save updated message history back to Redis (fast cache) and sync to PostgreSQL.
    5. Return AI response.
    """
    biz_id = str(current_user.business_id)

    # Validate conversation_id if provided
    if request.conversation_id:
        try:
            uuid.UUID(request.conversation_id)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Invalid conversation_id format, must be a valid UUID")

    conv_id = request.conversation_id or str(uuid.uuid4())

    # 1. Fetch Session Memory from Redis
    session_messages: List[Dict[str, Any]] = await redis_service.get_session_memory(biz_id, conv_id)

    # If Redis had expired or empty, try loading existing conversation from Postgres
    if not session_messages and request.conversation_id:
        try:
            conv_uuid = uuid.UUID(request.conversation_id)
            conv_stmt = select(Conversation).where(
                Conversation.id == conv_uuid,
                Conversation.business_id == current_user.business_id,
            )
            conv_obj = (await db.execute(conv_stmt)).scalars().first()
            if conv_obj and conv_obj.messages:
                session_messages = conv_obj.messages
        except Exception:
            pass

    # 2. Append new user message
    session_messages.append({"role": "user", "content": request.message})

    # Keep conversation sliding window bounded (last 10 turns)
    if len(session_messages) > 20:
        session_messages = session_messages[-20:]

    # 3. Generate response via LLM Adapter with live telemetry context
    llm_result = await llm_adapter.generate(
        messages=session_messages,
        business_id=biz_id,
        system_prompt=FRIDAY_SYSTEM_PROMPT,
        inject_telemetry=True,
    )

    assistant_content = llm_result.get("content", "Systems operational.")
    model_name = llm_result.get("model", "nvidia-nim")
    is_cached = llm_result.get("cached", False)

    # 4. Save response to session memory in Redis (24h TTL)
    session_messages.append({"role": "assistant", "content": assistant_content})
    await redis_service.save_session_memory(biz_id, conv_id, session_messages, ttl_seconds=86400)

    # 5. Asynchronously persist/update Conversation in Postgres
    try:
        conv_uuid = uuid.UUID(conv_id)
        conv_stmt = select(Conversation).where(Conversation.id == conv_uuid)
        existing_conv = (await db.execute(conv_stmt)).scalars().first()

        if existing_conv:
            existing_conv.messages = session_messages
        else:
            new_conv = Conversation(
                id=conv_uuid,
                business_id=current_user.business_id,
                user_id=current_user.id,
                title=request.message[:40] + ("..." if len(request.message) > 40 else ""),
                mode=request.mode or "chat",
                messages=session_messages,
            )
            db.add(new_conv)
        await db.commit()
    except Exception as e:
        # Non-blocking for sync chat path if DB sync encounters race
        logger.warning("Failed to persist conversation %s to DB: %s", conv_id, e)

    return FridayChatResponse(
        conversation_id=conv_id,
        response=assistant_content,
        model_used=model_name,
        tokens_used=llm_result.get("usage", {}).get("total_tokens", 0),
        cached=is_cached,
    )

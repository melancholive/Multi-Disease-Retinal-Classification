from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.react_agent import agent_graph


router = APIRouter(prefix="/agent", tags=["agent"])


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    message: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    # LangGraph message state can accept simple {role, content} dicts.
    messages: list[dict[str, Any]] = [m.model_dump() for m in body.messages]

    try:
        result: dict[str, Any] = await agent_graph.ainvoke({"messages": messages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    updated = result.get("messages") or []
    for m in reversed(updated):
        # LangChain messages typically provide `.type` and `.content`.
        content = getattr(m, "content", None)
        msg_type = getattr(m, "type", None)
        if msg_type in ("ai", "assistant") and isinstance(content, str) and content.strip():
            return ChatResponse(message=content)

        # Fallback for dict-like messages.
        if isinstance(m, dict) and m.get("role") in ("assistant", "ai"):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return ChatResponse(message=content)

    raise HTTPException(status_code=500, detail="agent returned no assistant message")

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.agent import RetinalAgentService


router = APIRouter(prefix="/agent", tags=["agent"])
agent_service = RetinalAgentService()


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str = Field(default="default-chat")


class ChatResponse(BaseModel):
    message: str


class AgentRequest(BaseModel):
    message: str
    session_id: str = Field(default="default")
    image_path: str | None = None
    top_diseases: list[dict[str, Any]] | list[str] | None = None
    top_k: int | None = None
    prediction_context: dict[str, Any] | None = None
    gradcam_images: dict[str, str] | str | None = None
    default_gradcam_model: str | None = None


class AgentRunResponse(BaseModel):
    answer: str
    messages: list[Any]


@router.post("", response_model=AgentRunResponse)
def run_agent(body: AgentRequest):
    try:
        return agent_service.run(
            message=body.message,
            session_id=body.session_id,
            image_path=body.image_path,
            top_diseases=body.top_diseases,
            top_k=body.top_k,
            prediction_context=body.prediction_context,
            gradcam_images=body.gradcam_images,
            default_gradcam_model=body.default_gradcam_model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    messages: list[dict[str, Any]] = [m.model_dump() for m in body.messages]
    config = {"configurable": {"thread_id": body.session_id}}

    try:
        result: dict[str, Any] = await agent_service.get_graph().ainvoke(
            {"messages": messages},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    updated = result.get("messages") or []
    for m in reversed(updated):
        content = getattr(m, "content", None)
        msg_type = getattr(m, "type", None)
        if msg_type in ("ai", "assistant") and isinstance(content, str) and content.strip():
            return ChatResponse(message=content)

        if isinstance(m, dict) and m.get("role") in ("assistant", "ai"):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return ChatResponse(message=content)

    raise HTTPException(status_code=500, detail="agent returned no assistant message")

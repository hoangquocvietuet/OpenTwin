"""Chat and export API endpoints. Thin wrappers around chat_service."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.chat_service import chat as chat_service_fn, ChatResult
from app.database import ChatMessage


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    content: str
    retrieval_metadata: dict | None = None


def create_chat_router(
    collection,
    session_factory,
    twin_slug: str,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        content = req.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="Please enter a message.")

        result = chat_service_fn(
            content=req.content,
            collection=collection,
            session_factory=session_factory,
            twin_slug=twin_slug,
            system_prompt=system_prompt,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
        return ChatResponse(
            content=result.content,
            retrieval_metadata=result.retrieval_metadata,
        )

    @router.get("/api/export")
    def export_chat():
        """Export chat history as JSON."""
        with session_factory() as session:
            messages = (
                session.query(ChatMessage)
                .filter_by(twin_slug=twin_slug)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "retrieval_metadata": msg.retrieval_metadata,
                }
                for msg in messages
            ]

    return router

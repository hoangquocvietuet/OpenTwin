"""V2 API router: conversations, streaming chat, settings, test-connection."""

import json
import logging
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func

from app.config import settings as default_settings
from app.database import (
    AppSetting,
    ChatMessage,
    Conversation,
    load_settings as load_db_settings,
    save_settings as save_db_settings,
)
from app.chat_service import chat_stream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateConversationRequest(BaseModel):
    title: str = "New Chat"


class SettingsUpdate(BaseModel):
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None


class TestConnectionRequest(BaseModel):
    base_url: str
    api_key: str


class ChatStreamRequest(BaseModel):
    content: str
    mode: str = "answer"
    conversation_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_api_v2_router(
    session_factory,
    twin_slug: str,
    twin_name: str,
    system_prompt: str,
    rewrite_prompt: str,
    chromadb_client,
    data_dir: str,
    collection,
) -> APIRouter:
    """Create and return the /api/v2 router."""

    router = APIRouter(prefix="/api/v2")

    # -----------------------------------------------------------------------
    # Conversations
    # -----------------------------------------------------------------------

    @router.get("/conversations")
    def list_conversations():
        """List all conversations for this twin, ordered by updated_at desc."""
        with session_factory() as session:
            last_msg = (
                session.query(
                    ChatMessage.conversation_id,
                    func.max(ChatMessage.id).label("max_id"),
                )
                .filter(ChatMessage.conversation_id.isnot(None))
                .group_by(ChatMessage.conversation_id)
                .subquery()
            )
            results = (
                session.query(Conversation, ChatMessage.content, ChatMessage.created_at)
                .outerjoin(last_msg, Conversation.id == last_msg.c.conversation_id)
                .outerjoin(ChatMessage, ChatMessage.id == last_msg.c.max_id)
                .filter(Conversation.twin_slug == twin_slug)
                .order_by(Conversation.updated_at.desc())
                .all()
            )
            return [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                    "last_message": content,
                    "last_message_at": msg_created_at.isoformat() if msg_created_at else None,
                }
                for conv, content, msg_created_at in results
            ]

    @router.post("/conversations")
    def create_conversation(req: CreateConversationRequest):
        """Create a new conversation."""
        with session_factory() as session:
            conv = Conversation(twin_slug=twin_slug, title=req.title)
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            }

    @router.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        """Delete a conversation by ID. Returns 404 if not found."""
        with session_factory() as session:
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            session.delete(conv)
            session.commit()
            return {"ok": True}

    @router.get("/conversations/{conversation_id}/messages")
    def get_messages(conversation_id: str, limit: int = 50, before_id: Optional[int] = None):
        """Get paginated messages for a conversation. Returns 404 if conversation not found."""
        with session_factory() as session:
            conv = session.query(Conversation).filter_by(id=conversation_id).first()
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Get the latest `limit` messages (by id desc), then reverse to asc for display
            q = (
                session.query(ChatMessage)
                .filter_by(conversation_id=conversation_id)
            )
            if before_id is not None:
                q = q.filter(ChatMessage.id < before_id)

            messages = (
                q.order_by(ChatMessage.id.desc())
                .limit(limit)
                .all()
            )
            # Return in ascending order
            messages = list(reversed(messages))
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "conversation_id": msg.conversation_id,
                }
                for msg in messages
            ]

    # -----------------------------------------------------------------------
    # Streaming chat
    # -----------------------------------------------------------------------

    @router.post("/chat/stream")
    def chat_stream_endpoint(req: ChatStreamRequest):
        """Streaming chat endpoint. Returns SSE."""
        # Load current settings from DB
        db_settings = load_db_settings(session_factory)
        llm_base_url = db_settings.get("llm_base_url", default_settings.llm_base_url)
        llm_model = db_settings.get("llm_model", default_settings.llm_model)
        llm_api_key = db_settings.get("llm_api_key", default_settings.llm_api_key)

        def generate():
            try:
                for item in chat_stream(
                    content=req.content,
                    collection=collection,
                    session_factory=session_factory,
                    twin_slug=twin_slug,
                    twin_name=twin_name,
                    system_prompt=system_prompt,
                    rewrite_prompt=rewrite_prompt,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                    mode=req.mode,
                    conversation_id=req.conversation_id,
                ):
                    if isinstance(item, str):
                        # Text chunk: Vercel AI SDK format 0:{json}\n
                        yield f"0:{json.dumps(item)}\n"
                    elif isinstance(item, dict):
                        # Metadata or error
                        yield f"data: {json.dumps(item)}\n\n"
            except Exception as exc:
                logger.error("chat_stream error: %s", exc, exc_info=True)
                yield f"data: {json.dumps({'error': True, 'content': str(exc)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # -----------------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------------

    @router.get("/settings")
    def get_settings():
        """Return current settings, merging DB overrides with defaults."""
        db_settings = load_db_settings(session_factory)
        return {
            "llm_base_url": db_settings.get("llm_base_url", default_settings.llm_base_url),
            "llm_model": db_settings.get("llm_model", default_settings.llm_model),
            "llm_api_key": db_settings.get("llm_api_key", default_settings.llm_api_key),
            "embedding_model": db_settings.get("embedding_model", default_settings.embedding_model),
            "embedding_base_url": db_settings.get("embedding_base_url", default_settings.embedding_base_url),
            "embedding_api_key": db_settings.get("embedding_api_key", default_settings.embedding_api_key),
        }

    @router.put("/settings")
    def put_settings(req: SettingsUpdate):
        """Update settings. Only non-null fields are saved."""
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if updates:
            save_db_settings(session_factory, updates)
        return get_settings()

    # -----------------------------------------------------------------------
    # Test connection
    # -----------------------------------------------------------------------

    @router.post("/test-connection")
    def test_connection(req: TestConnectionRequest):
        """Test connectivity to an LLM provider. Makes GET to {base_url}/models."""
        start = time.monotonic()
        try:
            response = httpx.get(
                f"{req.base_url}/models",
                headers={"Authorization": f"Bearer {req.api_key}"},
                timeout=5.0,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": True,
                "latency_ms": latency_ms,
                "status_code": response.status_code,
            }
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": str(exc),
            }

    # -----------------------------------------------------------------------
    # Sources (import history)
    # -----------------------------------------------------------------------

    @router.get("/sources")
    def list_sources():
        """List all imported sources for this twin."""
        from app.sources import load_sources
        sources = load_sources(data_dir, twin_slug)
        return [
            {
                "id": s.id,
                "name": s.name,
                "platform": s.platform,
                "enabled": s.enabled,
                "created_at": s.created_at,
                "total_messages": s.total_messages,
                "target_messages": s.target_messages,
                "train_chunks": s.train_chunks,
            }
            for s in sources
        ]

    @router.patch("/sources/{source_id}")
    def toggle_source_endpoint(source_id: str, enabled: bool = True):
        """Enable or disable a source."""
        from app.sources import toggle_source
        found = toggle_source(data_dir, twin_slug, source_id, enabled)
        if not found:
            raise HTTPException(status_code=404, detail="Source not found")
        return {"ok": True}

    @router.delete("/sources/{source_id}")
    def delete_source_endpoint(source_id: str):
        """Delete a source and remove its embeddings from ChromaDB."""
        from app.sources import delete_source
        from app.importer import remove_source_embeddings

        db_settings = load_db_settings(session_factory)
        emb_model = db_settings.get("embedding_model", "text-embedding-3-small")
        emb_base_url = db_settings.get("embedding_base_url", "http://localhost:11434/v1")
        emb_api_key = db_settings.get("embedding_api_key", "ollama")

        remove_source_embeddings(
            twin_slug=twin_slug,
            source_id=source_id,
            chromadb_client=chromadb_client,
            embedding_model=emb_model,
            embedding_base_url=emb_base_url,
            embedding_api_key=emb_api_key,
        )
        found = delete_source(data_dir, twin_slug, source_id)
        if not found:
            raise HTTPException(status_code=404, detail="Source not found")
        return {"ok": True}

    # -----------------------------------------------------------------------
    # Import
    # -----------------------------------------------------------------------

    from fastapi import UploadFile, File, Form

    @router.post("/import")
    async def import_data(
        file: UploadFile = File(...),
        source_name: str = Form(""),
        target_name: str = Form(""),
    ):
        """Import a zip file. Streams progress via SSE, final event is the result."""
        import tempfile, shutil, os, queue, threading
        from app.importer import run_import_pipeline, ZipValidationError

        db_settings = load_db_settings(session_factory)
        emb_model = db_settings.get("embedding_model", "text-embedding-3-small")
        emb_base_url = db_settings.get("embedding_base_url", "http://localhost:11434/v1")
        emb_api_key = db_settings.get("embedding_api_key", "ollama")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        progress_queue: queue.Queue = queue.Queue()

        def on_progress(msg: str):
            progress_queue.put(("progress", msg))

        def run_import():
            try:
                result = run_import_pipeline(
                    zip_path=tmp_path,
                    chromadb_client=chromadb_client,
                    data_dir=data_dir,
                    embedding_model=emb_model,
                    target_name=target_name or None,
                    source_name=source_name or "",
                    on_progress=on_progress,
                    embedding_base_url=emb_base_url,
                    embedding_api_key=emb_api_key,
                )
                progress_queue.put(("done", {"ok": True, **result}))
            except (ZipValidationError, ValueError) as e:
                progress_queue.put(("error", str(e)))
            except Exception:
                logger.exception("Import failed")
                progress_queue.put(("error", "Import failed"))
            finally:
                os.unlink(tmp_path)

        threading.Thread(target=run_import, daemon=True).start()

        def generate():
            while True:
                try:
                    kind, payload = progress_queue.get(timeout=300)
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Import timed out'})}\n\n"
                    return
                if kind == "progress":
                    yield f"data: {json.dumps({'type': 'progress', 'message': payload})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'type': 'done', **payload})}\n\n"
                    return
                elif kind == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                    return

        return StreamingResponse(generate(), media_type="text/event-stream")

    return router

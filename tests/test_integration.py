"""Integration test: full pipeline from chunks to chat response."""

import json
import os
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat import create_chat_router
from app.database import create_engine_and_tables, SessionFactory, ChatMessage
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function
from app.prompt import build_system_prompt, load_fingerprint
from app.retrieval import retrieve_chunks


def test_full_chat_pipeline(tmp_data_dir, tmp_path):
    """End-to-end: ingest chunks -> build prompt -> query -> get response."""
    # 1. Setup DB with session factory
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    session_factory = SessionFactory(engine)

    # 2. Load fingerprint and build system prompt
    fp_path = str(tmp_data_dir / "hoang_quoc_viet" / "style_fingerprint.json")
    fp = load_fingerprint(fp_path)
    system_prompt = build_system_prompt("Hoàng Quốc Việt", fp)
    assert "Hoàng Quốc Việt" in system_prompt

    # 3. Ingest chunks into ChromaDB with explicit embedding function
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")
    chunks_path = str(tmp_data_dir / "hoang_quoc_viet" / "train_chunks.jsonl")
    chunks = load_chunks_from_jsonl(chunks_path)
    collection = ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)
    assert collection.count() == 3

    # 4. Verify retrieval works
    retrieved = retrieve_chunks(collection, "đang làm gì", n_results=2)
    assert len(retrieved) > 0

    # 5. Test chat endpoint with mocked LLM
    app = FastAPI()
    router = create_chat_router(
        collection=collection,
        session_factory=session_factory,
        twin_slug="hoang_quoc_viet",
        system_prompt=system_prompt,
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    app.include_router(router)

    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "đang code nè bạn"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        test_client = TestClient(app)
        resp = test_client.post("/api/chat", json={"content": "bạn đang làm gì thế?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "đang code nè bạn"
        assert body["retrieval_metadata"]["chunks"] > 0

    # 6. Verify chat was saved to DB
    with session_factory() as session:
        messages = session.query(ChatMessage).all()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].twin_slug == "hoang_quoc_viet"
        assert messages[1].role == "assistant"
        assert messages[1].content == "đang code nè bạn"

    # 7. Test export endpoint
    test_client2 = TestClient(app)
    resp = test_client2.get("/api/export")
    assert resp.status_code == 200
    exported = resp.json()
    assert len(exported) == 2
    assert exported[0]["role"] == "user"
    assert exported[1]["role"] == "assistant"

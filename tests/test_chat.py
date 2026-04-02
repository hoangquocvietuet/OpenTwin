import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat import create_chat_router


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection that returns test results."""
    collection = MagicMock()
    collection.count.return_value = 3
    collection.query.return_value = {
        "ids": [["dm_test_0"]],
        "documents": [["Friend: bạn làm gì?\nViệt: đang code nè"]],
        "distances": [[0.3]],
        "metadatas": [[{"chunk_type": "dm", "score": 1.5}]],
    }
    return collection


@pytest.fixture
def mock_session_factory():
    """Mock session factory."""
    session = MagicMock()
    session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
    session.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []

    factory = MagicMock()
    factory.return_value.__enter__ = MagicMock(return_value=session)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    return factory


@pytest.fixture
def chat_app(mock_collection, mock_session_factory):
    """FastAPI app with chat router for testing."""
    app = FastAPI()
    router = create_chat_router(
        collection=mock_collection,
        session_factory=mock_session_factory,
        twin_slug="hoang_quoc_viet",
        twin_name="Việt",
        system_prompt="You are Việt. Respond casually.",
        rewrite_prompt="Rephrase in your style.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    app.include_router(router)
    return app


def test_chat_rejects_empty_message(chat_app):
    """POST /api/chat with empty content returns 400."""
    client = TestClient(chat_app)
    resp = client.post("/api/chat", json={"content": ""})
    assert resp.status_code == 400
    assert "message" in resp.json()["detail"].lower()


def test_chat_returns_no_twin_error_when_empty_collection(mock_session_factory):
    """When ChromaDB collection is empty, returns appropriate error."""
    empty_collection = MagicMock()
    empty_collection.count.return_value = 0

    app = FastAPI()
    router = create_chat_router(
        collection=empty_collection,
        session_factory=mock_session_factory,
        twin_slug="hoang_quoc_viet",
        twin_name="Việt",
        system_prompt="You are Việt.",
        rewrite_prompt="Rephrase in your style.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"content": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert "not enough context" in body["content"].lower() or "import data" in body["content"].lower()


def test_chat_success_with_mocked_llm(chat_app, mock_collection):
    """Successful chat returns LLM response with retrieval metadata."""
    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "đang code nè bạn"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        client = TestClient(chat_app)
        resp = client.post("/api/chat", json={"content": "bạn đang làm gì?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "đang code nè bạn"
        assert body["retrieval_metadata"]["chunks"] > 0


def test_export_empty_history(chat_app):
    """GET /api/export with no messages returns empty list."""
    client = TestClient(chat_app)
    resp = client.get("/api/export")
    assert resp.status_code == 200
    assert resp.json() == []

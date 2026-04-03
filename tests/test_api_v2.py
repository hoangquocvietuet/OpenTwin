"""Tests for v2 API endpoints (conversations, streaming, settings, test-connection)."""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.database import create_engine_and_tables, SessionFactory, Conversation, ChatMessage


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine_and_tables(str(tmp_path / "test.db"))
    return SessionFactory(engine)


@pytest.fixture
def app(session_factory):
    from fastapi import FastAPI
    from app.api_v2 import create_api_v2_router

    app = FastAPI()
    router = create_api_v2_router(
        session_factory=session_factory,
        twin_slug="test_twin",
        twin_name="Test Twin",
        system_prompt="You are a test twin.",
        rewrite_prompt="Rewrite in test voice.",
        chromadb_client=MagicMock(),
        data_dir="/tmp/test",
        collection=MagicMock(),
    )
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestConversations:
    def test_create_conversation(self, client):
        resp = client.post("/api/v2/conversations", json={"title": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Hello"
        assert "id" in data

    def test_list_conversations_empty(self, client):
        resp = client.get("/api/v2/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_conversations_with_messages(self, client, session_factory):
        conv_id = str(uuid.uuid4())
        with session_factory() as session:
            session.add(Conversation(id=conv_id, twin_slug="test_twin", title="Test"))
            session.add(ChatMessage(twin_slug="test_twin", role="user", content="hi there", conversation_id=conv_id))
            session.commit()

        resp = client.get("/api/v2/conversations")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == conv_id
        assert data[0]["last_message"] == "hi there"

    def test_delete_conversation(self, client, session_factory):
        conv_id = str(uuid.uuid4())
        with session_factory() as session:
            session.add(Conversation(id=conv_id, twin_slug="test_twin", title="Test"))
            session.commit()

        resp = client.delete(f"/api/v2/conversations/{conv_id}")
        assert resp.status_code == 200

        resp = client.get("/api/v2/conversations")
        assert resp.json() == []

    def test_delete_nonexistent_conversation(self, client):
        resp = client.delete("/api/v2/conversations/nonexistent")
        assert resp.status_code == 404

    def test_get_messages_with_pagination(self, client, session_factory):
        conv_id = str(uuid.uuid4())
        with session_factory() as session:
            session.add(Conversation(id=conv_id, twin_slug="test_twin", title="Test"))
            for i in range(5):
                session.add(ChatMessage(
                    twin_slug="test_twin", role="user",
                    content=f"msg {i}", conversation_id=conv_id,
                ))
            session.commit()

        # Get latest 3
        resp = client.get(f"/api/v2/conversations/{conv_id}/messages?limit=3")
        data = resp.json()
        assert len(data) == 3
        # Should be latest messages (msg 4, msg 3, msg 2) in ascending order
        assert data[-1]["content"] == "msg 4"

    def test_get_messages_nonexistent_conversation(self, client):
        resp = client.get("/api/v2/conversations/nonexistent/messages")
        assert resp.status_code == 404


class TestSettings:
    def test_get_settings(self, client):
        resp = client.get("/api/v2/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_base_url" in data

    def test_put_settings(self, client):
        resp = client.put("/api/v2/settings", json={
            "llm_base_url": "http://new-url:11434/v1",
            "llm_model": "new-model",
        })
        assert resp.status_code == 200

        resp = client.get("/api/v2/settings")
        data = resp.json()
        assert data["llm_base_url"] == "http://new-url:11434/v1"
        assert data["llm_model"] == "new-model"


class TestTestConnection:
    def test_connection_success(self, client):
        with patch("app.api_v2.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.get.return_value = mock_resp

            resp = client.post("/api/v2/test-connection", json={
                "base_url": "http://localhost:11434/v1",
                "api_key": "test",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True
            assert "latency_ms" in data

    def test_connection_failure(self, client):
        with patch("app.api_v2.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("Connection refused")

            resp = client.post("/api/v2/test-connection", json={
                "base_url": "http://localhost:99999/v1",
                "api_key": "test",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is False

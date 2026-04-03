# tests/test_chat_service_stream.py
"""Tests for streaming chat service."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.chat_service import chat_stream


@pytest.fixture
def mock_collection():
    coll = MagicMock()
    coll.count.return_value = 10
    coll.query.return_value = {
        "ids": [["chunk_1"]],
        "documents": [["Friend: hey\nTwin: yo whats up"]],
        "distances": [[0.3]],
        "metadatas": [[{"chunk_type": "dm", "quality_score": 1.0}]],
    }
    return coll


@pytest.fixture
def mock_session_factory(tmp_path):
    from app.database import create_engine_and_tables, SessionFactory
    engine = create_engine_and_tables(str(tmp_path / "test.db"))
    return SessionFactory(engine)


def test_chat_stream_yields_text_then_metadata(mock_collection, mock_session_factory):
    """chat_stream should yield text chunks, then a final metadata dict."""
    with patch("app.chat_service._has_enriched_metadata", return_value=False):
        with patch("app.chat_service.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "yo whats up dude"
            mock_response.usage = MagicMock(total_tokens=50)
            mock_client.chat.completions.create.return_value = mock_response

            chunks = list(chat_stream(
                content="hello",
                collection=mock_collection,
                session_factory=mock_session_factory,
                twin_slug="test",
                twin_name="Test Twin",
                system_prompt="You are a twin.",
                rewrite_prompt="Rewrite.",
                llm_base_url="http://localhost:11434/v1",
                llm_model="test-model",
                llm_api_key="test-key",
                mode="answer",
            ))

            # Last item should be metadata dict
            assert len(chunks) >= 2
            # Text chunks are strings
            text_chunks = [c for c in chunks if isinstance(c, str)]
            assert len(text_chunks) > 0
            assert "".join(text_chunks) == "yo whats up dude"

            # Last item is metadata
            metadata = chunks[-1]
            assert isinstance(metadata, dict)
            assert "chunks" in metadata
            assert "avg_similarity" in metadata


def test_chat_stream_empty_content():
    """Empty content should yield an error."""
    chunks = list(chat_stream(
        content="",
        collection=MagicMock(),
        session_factory=MagicMock(),
        twin_slug="test",
        twin_name="Test",
        system_prompt="",
        rewrite_prompt="",
        llm_base_url="",
        llm_model="",
        llm_api_key="",
    ))
    assert len(chunks) == 1
    assert isinstance(chunks[0], dict)
    assert chunks[0].get("error") is True

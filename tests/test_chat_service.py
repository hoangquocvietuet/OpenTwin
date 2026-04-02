import json
from unittest.mock import MagicMock, patch

import pytest

from app.chat_service import chat, ChatResult


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
    """Mock session factory that returns a mock session."""
    session = MagicMock()
    session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

    factory = MagicMock()
    factory.return_value.__enter__ = MagicMock(return_value=session)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    return factory


def test_chat_rejects_empty_message(mock_collection, mock_session_factory):
    """Empty content returns error result."""
    result = chat(
        content="",
        collection=mock_collection,
        session_factory=mock_session_factory,
        twin_slug="test",
        twin_name="Việt",
        system_prompt="You are Việt.",
        rewrite_prompt="Rephrase in your style.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    assert result.error is True
    assert "message" in result.content.lower()


def test_chat_returns_no_data_when_empty_collection(mock_session_factory):
    """When collection is empty, returns import prompt."""
    empty_collection = MagicMock()
    empty_collection.count.return_value = 0

    result = chat(
        content="hello",
        collection=empty_collection,
        session_factory=mock_session_factory,
        twin_slug="test",
        twin_name="Việt",
        system_prompt="You are Việt.",
        rewrite_prompt="Rephrase in your style.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    assert "import" in result.content.lower() or "data" in result.content.lower()


def test_chat_truncates_and_calls_llm(mock_collection, mock_session_factory):
    """Long messages are truncated; LLM is called with truncated content."""
    long_msg = "a" * 15000

    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "đang code nè"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        result = chat(
            content=long_msg,
            collection=mock_collection,
            session_factory=mock_session_factory,
            twin_slug="test",
            twin_name="Việt",
            system_prompt="You are Việt.",
            rewrite_prompt="Rephrase in your style.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
            llm_api_key="ollama",
        )

        assert result.content == "đang code nè"
        assert result.error is False

        # Verify the user message in the LLM call was truncated to 10K
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][-1]
        assert len(user_msg["content"]) == 10_000


def test_chat_handles_connection_error(mock_collection, mock_session_factory):
    """Connection error returns friendly message."""
    import openai as openai_lib
    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai_lib.APIConnectionError(
            request=MagicMock()
        )

        result = chat(
            content="hello",
            collection=mock_collection,
            session_factory=mock_session_factory,
            twin_slug="test",
            twin_name="Việt",
            system_prompt="You are Việt.",
            rewrite_prompt="Rephrase in your style.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
            llm_api_key="ollama",
        )
        assert "ollama" in result.content.lower() or "reach" in result.content.lower()
        assert result.error is True


def test_chat_handles_json_decode_error(mock_collection, mock_session_factory):
    """Malformed LLM JSON response returns friendly message."""
    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = json.JSONDecodeError(
            "Expecting value", "", 0
        )

        result = chat(
            content="hello",
            collection=mock_collection,
            session_factory=mock_session_factory,
            twin_slug="test",
            twin_name="Việt",
            system_prompt="You are Việt.",
            rewrite_prompt="Rephrase in your style.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
            llm_api_key="ollama",
        )
        assert "unexpected" in result.content.lower() or "format" in result.content.lower()
        assert result.error is True

# tests/test_agent_intent.py

import re
from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.intent import intent_agent


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_intent_agent_casual_chat():
    """Short casual message classified correctly."""
    client = _make_mock_llm_client('{"intent": "casual_chat", "tone": "casual_banter"}')
    state = PipelineState(raw_input="tối nay ăn j", mode="answer")

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.intent == "casual_chat"
    assert result.tone == "casual_banter"
    assert result.needs_context is False
    assert result.resolved_content == "tối nay ăn j"


def test_intent_agent_long_text_no_context_needed():
    """Long text input doesn't need external context."""
    long_text = "VKSND TP.HCM đã ban hành cáo trạng " * 20
    client = _make_mock_llm_client('{"intent": "rewrite_article", "tone": "formal_news"}')
    state = PipelineState(raw_input=long_text, mode="rewrite")

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.needs_context is False
    assert result.resolved_content == long_text


def test_intent_agent_url_needs_context():
    """Message with URL triggers context fetching."""
    client = _make_mock_llm_client('{"intent": "rewrite_article", "tone": "formal_news"}')
    state = PipelineState(
        raw_input="rewrite this https://vnexpress.net/article-123",
        mode="rewrite",
    )

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.needs_context is True
    assert result.context_source == "url"
    assert "vnexpress.net" in result.context_url


def test_intent_agent_handles_malformed_json():
    """Falls back to defaults when LLM returns bad JSON."""
    client = _make_mock_llm_client("not json at all")
    state = PipelineState(raw_input="hello", mode="answer")

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.intent == "general"
    assert result.tone == "neutral"
    assert result.resolved_content == "hello"

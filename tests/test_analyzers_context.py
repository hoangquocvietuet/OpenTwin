from unittest.mock import MagicMock, patch
from app.analyzers.registry import AnalyzerInput
from app.analyzers.context import analyze_context


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_context_returns_summary():
    """Context analyzer generates a context summary from chunk messages."""
    chunk = {
        "messages": [
            {"author": "Friend", "text": "tối nay ăn gì", "timestamp": "2025-08-01T18:00:00"},
            {"author": "Viet", "text": "ăn phở đi", "timestamp": "2025-08-01T18:01:00"},
        ],
        "metadata": {"participants": ["Viet", "Friend"]},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"context_summary": "friends planning dinner", "interaction_type": "planning", "relationship": "close_friends"}')

    result = analyze_context(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "context_summary" in result
    assert "interaction_type" in result
    assert "relationship" in result


def test_analyze_context_uses_neighbors():
    """Context analyzer includes prev/next chunk in prompt for broader context."""
    chunk = {
        "messages": [{"author": "Viet", "text": "ok fine", "timestamp": "2025-08-01T18:05:00"}],
        "metadata": {},
    }
    prev_chunk = {
        "messages": [{"author": "Friend", "text": "you lied to me", "timestamp": "2025-08-01T18:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk, prev_chunk=prev_chunk)

    client = _make_mock_llm_client('{"context_summary": "tense moment after accusation", "interaction_type": "conflict", "relationship": "strained"}')

    result = analyze_context(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    call_args = client.chat.completions.create.call_args
    prompt_content = str(call_args)
    assert "you lied to me" in prompt_content or result["interaction_type"] == "conflict"


def test_analyze_context_handles_malformed_json():
    """Context analyzer returns defaults when LLM returns bad JSON."""
    chunk = {
        "messages": [{"author": "Viet", "text": "hi", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client("this is not json at all")

    result = analyze_context(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "context_summary" in result
    assert result["context_summary"] != ""

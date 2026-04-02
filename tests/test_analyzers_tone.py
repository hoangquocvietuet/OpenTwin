from unittest.mock import MagicMock
from app.analyzers.registry import AnalyzerInput
from app.analyzers.tone import analyze_tone


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_tone_returns_fields():
    """Tone analyzer returns tone, formality, and energy."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "ê đi ăn k", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ok đi", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "friends planning to eat"},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client('{"tone": "casual_banter", "formality": 0.1, "energy": "relaxed"}')
    result = analyze_tone(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "tone" in result
    assert "formality" in result
    assert "energy" in result
    assert isinstance(result["formality"], (int, float))


def test_analyze_tone_uses_context_summary():
    """Tone analyzer includes context_summary from prior analyzer in its prompt."""
    chunk = {
        "messages": [{"author": "Viet", "text": "...", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {"context_summary": "heated argument about money"},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client('{"tone": "angry", "formality": 0.3, "energy": "high"}')
    result = analyze_tone(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    call_args = client.chat.completions.create.call_args
    prompt_content = str(call_args)
    assert "heated argument" in prompt_content


def test_analyze_tone_handles_malformed_json():
    """Tone analyzer returns defaults when LLM returns bad JSON."""
    chunk = {
        "messages": [{"author": "Viet", "text": "hi", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client("not json")
    result = analyze_tone(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["tone"] == "neutral"
    assert result["formality"] == 0.5

from unittest.mock import MagicMock
from app.analyzers.registry import AnalyzerInput
from app.analyzers.persona import analyze_persona


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_persona_returns_fields():
    """Persona analyzer returns twin_role, register, relationship_to_others."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "đi ăn k mấy ông", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ok đi", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "friend group planning dinner"},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client('{"twin_role": "initiator", "register": "informal_close", "relationship_to_others": "friend_banter"}')
    result = analyze_persona(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["twin_role"] == "initiator"
    assert result["register"] == "informal_close"
    assert result["relationship_to_others"] == "friend_banter"


def test_analyze_persona_handles_malformed_json():
    """Persona analyzer returns defaults on bad JSON."""
    chunk = {
        "messages": [{"author": "Viet", "text": "hi", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client("garbage response")
    result = analyze_persona(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["twin_role"] == "participant"
    assert result["register"] == "unknown"

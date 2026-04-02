from unittest.mock import MagicMock
from app.analyzers.registry import AnalyzerInput
from app.analyzers.emotion import analyze_emotion


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_emotion_returns_fields():
    """Emotion analyzer returns emotion, sentiment, conflict, sarcasm."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "haha bạn ngu quá", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "🤣🤣🤣", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "friends joking around"},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client('{"emotion": "playful", "sentiment": 0.8, "conflict": false, "sarcasm": false}')
    result = analyze_emotion(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "emotion" in result
    assert "sentiment" in result
    assert "conflict" in result
    assert "sarcasm" in result
    assert isinstance(result["conflict"], bool)


def test_analyze_emotion_detects_conflict():
    """Emotion analyzer identifies conflict in arguments."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "sao m nói vậy", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "thì sự thật mà", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "disagreement about plans"},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client('{"emotion": "frustrated", "sentiment": -0.3, "conflict": true, "sarcasm": false}')
    result = analyze_emotion(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["conflict"] is True
    assert result["sentiment"] < 0

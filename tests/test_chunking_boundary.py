from unittest.mock import MagicMock
from app.chunking.boundary import detect_boundaries


def _make_mock_llm_client(responses: list[str]):
    """Mock LLM that returns different responses for sequential calls."""
    client = MagicMock()
    mock_responses = []
    for text in responses:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = text
        mock_responses.append(mock_response)
    client.chat.completions.create.side_effect = mock_responses
    return client


def test_detect_boundaries_splits_on_topic_shift():
    """Boundary detection splits when LLM identifies topic shift."""
    messages = [
        {"author": "A", "text": "đi ăn k", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "ok đi", "timestamp": "2025-08-01T10:01:00"},
        {"author": "A", "text": "ăn phở nhé", "timestamp": "2025-08-01T10:02:00"},
        {"author": "A", "text": "ê m làm xong bài chưa", "timestamp": "2025-08-01T10:30:00"},
        {"author": "B", "text": "chưa", "timestamp": "2025-08-01T10:31:00"},
        {"author": "A", "text": "deadline mai rồi", "timestamp": "2025-08-01T10:32:00"},
    ]

    client = _make_mock_llm_client([
        '{"boundaries": [3]}',
    ])

    boundaries = detect_boundaries(messages, llm_client=client, llm_model="test")

    assert 3 in boundaries


def test_detect_boundaries_no_split_on_short_conversation():
    """Short conversations (< 3 messages) get no internal boundaries."""
    messages = [
        {"author": "A", "text": "hi", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
    ]

    client = _make_mock_llm_client(['{"boundaries": []}'])

    boundaries = detect_boundaries(messages, llm_client=client, llm_model="test")

    assert boundaries == []


def test_detect_boundaries_returns_empty_without_llm():
    """Without LLM client, returns empty boundaries (single chunk)."""
    messages = [
        {"author": "A", "text": "hi", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
    ]

    boundaries = detect_boundaries(messages, llm_client=None, llm_model=None)

    assert boundaries == []

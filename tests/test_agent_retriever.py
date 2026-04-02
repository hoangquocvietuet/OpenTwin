# tests/test_agent_retriever.py

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.retriever import retriever_agent


def _make_mock_collection(tone_results=None, content_results=None):
    """Mock ChromaDB collection that returns different results per query."""
    collection = MagicMock()
    collection.count.return_value = 100

    call_count = [0]
    def mock_query(**kwargs):
        call_count[0] += 1
        where = kwargs.get("where")
        # First call = tone query (has tone filter), second = content query
        if where and "$and" in str(where):
            return tone_results or {
                "ids": [["tone_1", "tone_2", "tone_3"]],
                "documents": [["casual doc 1", "casual doc 2", "casual doc 3"]],
                "distances": [[0.2, 0.3, 0.4]],
                "metadatas": [[
                    {"tone": "casual", "formality": 0.2, "twin_msg_ratio": 0.5},
                    {"tone": "casual_banter", "formality": 0.1, "twin_msg_ratio": 0.6},
                    {"tone": "playful", "formality": 0.3, "twin_msg_ratio": 0.4},
                ]],
            }
        else:
            return content_results or {
                "ids": [["content_1", "content_2"]],
                "documents": [["food discussion", "dinner plan"]],
                "distances": [[0.3, 0.8]],  # second one is low similarity
                "metadatas": [[
                    {"tone": "casual", "twin_msg_ratio": 0.5},
                    {"tone": "casual", "twin_msg_ratio": 0.3},
                ]],
            }

    collection.query = mock_query
    return collection


def test_retriever_agent_returns_tone_and_content_chunks():
    """Retriever returns both tone-matched and content-matched chunks."""
    collection = _make_mock_collection()
    state = PipelineState(
        raw_input="tối nay ăn j",
        mode="answer",
        intent="casual_chat",
        tone="casual_banter",
        resolved_content="tối nay ăn j",
    )

    result = retriever_agent(state, collection=collection)

    assert len(result.tone_chunks) > 0
    assert len(result.tone_chunks) <= 3
    assert len(result.content_chunks) >= 0
    assert len(result.content_chunks) <= 2


def test_retriever_agent_filters_low_similarity_content():
    """Content chunks with similarity below threshold are excluded."""
    # All content results have high distance (low similarity)
    content_results = {
        "ids": [["c1", "c2"]],
        "documents": [["irrelevant 1", "irrelevant 2"]],
        "distances": [[0.9, 0.95]],  # very dissimilar
        "metadatas": [[{"twin_msg_ratio": 0.5}, {"twin_msg_ratio": 0.5}]],
    }
    collection = _make_mock_collection(content_results=content_results)
    state = PipelineState(
        raw_input="something unrelated",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="something unrelated",
    )

    result = retriever_agent(state, collection=collection)

    assert result.content_chunks == []


def test_retriever_agent_handles_empty_collection():
    """Retriever handles empty collection gracefully."""
    collection = MagicMock()
    collection.count.return_value = 0

    state = PipelineState(
        raw_input="hello",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="hello",
    )

    result = retriever_agent(state, collection=collection)

    assert result.tone_chunks == []
    assert result.content_chunks == []

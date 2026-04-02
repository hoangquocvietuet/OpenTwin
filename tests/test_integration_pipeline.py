"""End-to-end: enriched chunks → pipeline → response with critic loop."""

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.graph import run_pipeline


def test_full_pipeline_answer_mode():
    """Full pipeline produces an approved answer response."""
    # Mock collection with enriched metadata
    collection = MagicMock()
    collection.count.return_value = 10
    collection.query.return_value = {
        "ids": [["c1", "c2", "c3"]],
        "documents": [["Viet: đang code nè\nFriend: dự án gì", "Viet: ăn phở đi", "Viet: ok"]],
        "distances": [[0.2, 0.3, 0.4]],
        "metadatas": [[
            {"tone": "casual", "formality": 0.2, "twin_msg_ratio": 0.6},
            {"tone": "casual_banter", "formality": 0.1, "twin_msg_ratio": 0.5},
            {"tone": "casual", "formality": 0.3, "twin_msg_ratio": 1.0},
        ]],
    }

    def mock_create(**kwargs):
        msgs = kwargs.get("messages", [])
        system_content = msgs[0]["content"] if msgs else ""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]

        if "classify" in system_content.lower():
            mock_resp.choices[0].message.content = '{"intent": "casual_chat", "tone": "casual_banter"}'
        elif "quality reviewer" in system_content.lower():
            mock_resp.choices[0].message.content = '{"approved": true, "feedback": ""}'
        else:
            mock_resp.choices[0].message.content = "đang code nè bạn"
        return mock_resp

    llm_client = MagicMock()
    llm_client.chat.completions.create = mock_create

    result = run_pipeline(
        raw_input="đang làm gì",
        mode="answer",
        collection=collection,
        llm_client=llm_client,
        llm_model="test",
        classifier_client=llm_client,
        classifier_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase.",
    )

    assert result.approved is True
    assert result.draft_response == "đang code nè bạn"
    assert result.intent == "casual_chat"
    assert result.tone == "casual_banter"
    assert len(result.tone_chunks) > 0


def test_full_pipeline_rewrite_with_critic_retry():
    """Rewrite mode: critic rejects answer-style response, retries succeed."""
    collection = MagicMock()
    collection.count.return_value = 10
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["Viet: tối nay ăn j"]],
        "distances": [[0.3]],
        "metadatas": [[{"tone": "casual", "formality": 0.2, "twin_msg_ratio": 0.5}]],
    }

    attempt = [0]

    def mock_create(**kwargs):
        msgs = kwargs.get("messages", [])
        system_content = msgs[0]["content"] if msgs else ""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]

        if "classify" in system_content.lower():
            mock_resp.choices[0].message.content = '{"intent": "rewrite_casual", "tone": "casual"}'
        elif "quality reviewer" in system_content.lower():
            attempt[0] += 1
            if attempt[0] <= 1:
                mock_resp.choices[0].message.content = '{"approved": false, "feedback": "You answered instead of rephrasing."}'
            else:
                mock_resp.choices[0].message.content = '{"approved": true, "feedback": ""}'
        else:
            mock_resp.choices[0].message.content = "tối nay ăn j đây"
        return mock_resp

    llm_client = MagicMock()
    llm_client.chat.completions.create = mock_create

    result = run_pipeline(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        collection=collection,
        llm_client=llm_client,
        llm_model="test",
        classifier_client=llm_client,
        classifier_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase in your style.",
    )

    assert result.approved is True
    assert result.retry_count >= 1
    assert result.draft_response == "tối nay ăn j đây"

# tests/test_pipeline_graph.py

from unittest.mock import MagicMock, patch
from app.pipeline.graph import build_pipeline, run_pipeline


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_build_pipeline_returns_compiled_graph():
    """build_pipeline returns a compiled LangGraph with bound dependencies."""
    pipeline = build_pipeline()
    assert pipeline is not None


def test_run_pipeline_answer_mode():
    """Full pipeline run in answer mode produces a response."""
    collection = MagicMock()
    collection.count.return_value = 5
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["Viet: đang code"]],
        "distances": [[0.3]],
        "metadatas": [[{"tone": "casual", "twin_msg_ratio": 0.5}]],
    }

    # Intent: casual_chat
    # Critic: approved
    intent_resp = '{"intent": "casual_chat", "tone": "casual_banter"}'
    critic_resp = '{"approved": true, "feedback": ""}'
    responder_resp = "đang code nè"

    call_count = [0]
    def mock_create(**kwargs):
        call_count[0] += 1
        msgs = kwargs.get("messages", [])
        system_content = msgs[0]["content"] if msgs else ""

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]

        # Route based on system prompt content
        if "classify" in system_content.lower():
            mock_resp.choices[0].message.content = intent_resp
        elif "quality reviewer" in system_content.lower():
            mock_resp.choices[0].message.content = critic_resp
        else:
            mock_resp.choices[0].message.content = responder_resp

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

    assert result.draft_response == "đang code nè"
    assert result.approved is True


def test_run_pipeline_critic_retry():
    """Pipeline retries when critic rejects, then approves on second try."""
    collection = MagicMock()
    collection.count.return_value = 5
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["Viet: ăn phở"]],
        "distances": [[0.3]],
        "metadatas": [[{"tone": "casual", "twin_msg_ratio": 0.5}]],
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
        rewrite_prompt="Rephrase.",
    )

    assert result.approved is True
    assert result.retry_count >= 1

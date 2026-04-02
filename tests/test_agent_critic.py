# tests/test_agent_critic.py

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.critic import critic_agent


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_critic_approves_good_response():
    """Critic approves when response matches tone and mode."""
    client = _make_mock_llm_client('{"approved": true, "feedback": ""}')
    state = PipelineState(
        raw_input="tối nay ăn j",
        mode="answer",
        intent="casual_chat",
        tone="casual_banter",
        resolved_content="tối nay ăn j",
        tone_chunks=[{"document": "Viet: ăn phở đi"}],
        content_chunks=[],
        draft_response="ăn phở đi",
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    assert result.approved is True
    assert result.retry_count == 0


def test_critic_rejects_answer_in_rewrite_mode():
    """Critic rejects when rewrite mode got an answer instead of rephrase."""
    client = _make_mock_llm_client('{"approved": false, "feedback": "You answered the question instead of rephrasing it. The input is a question and the output should be a question in your style."}')
    state = PipelineState(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        intent="rewrite_casual",
        tone="casual",
        resolved_content="Tối nay mình ăn gì nhỉ",
        tone_chunks=[],
        content_chunks=[],
        draft_response="tối nay ăn ốc",  # answered instead of rephrased
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    assert result.approved is False
    assert result.critic_feedback is not None
    assert len(result.critic_feedback) > 0
    assert result.retry_count == 1


def test_critic_increments_retry_count():
    """Each rejection increments retry_count."""
    client = _make_mock_llm_client('{"approved": false, "feedback": "Wrong tone."}')
    state = PipelineState(
        raw_input="test",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="test",
        draft_response="I would be happy to help you!",
        retry_count=1,  # already retried once
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    assert result.approved is False
    assert result.retry_count == 2


def test_critic_handles_malformed_json():
    """Critic rejects on malformed LLM response (fail-safe)."""
    client = _make_mock_llm_client("not json")
    state = PipelineState(
        raw_input="test",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="test",
        draft_response="ok",
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    # Fail-safe: reject so bad output doesn't slip through
    assert result.approved is False
    assert result.retry_count == 1
    assert "unparseable" in result.critic_feedback.lower()

# tests/test_agent_responder.py

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.responder import responder_agent


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_responder_answer_mode():
    """Responder generates answer using system prompt and chunks."""
    client = _make_mock_llm_client("đang code nè")
    state = PipelineState(
        raw_input="đang làm gì",
        mode="answer",
        intent="casual_chat",
        tone="casual_banter",
        resolved_content="đang làm gì",
        tone_chunks=[{"document": "Viet: đang code dự án\nFriend: dự án gì"}],
        content_chunks=[],
    )

    result = responder_agent(
        state,
        llm_client=client,
        llm_model="test",
        system_prompt="You are Viet. Chat casually.",
        rewrite_prompt="Rephrase in your style.",
    )

    assert result.draft_response == "đang code nè"


def test_responder_rewrite_mode():
    """Responder uses rewrite prompt in rewrite mode."""
    client = _make_mock_llm_client("tối nay ăn j đây")
    state = PipelineState(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        intent="rewrite_casual",
        tone="casual",
        resolved_content="Tối nay mình ăn gì nhỉ",
        tone_chunks=[{"document": "Viet: ê ăn j\nFriend: phở"}],
        content_chunks=[],
    )

    result = responder_agent(
        state,
        llm_client=client,
        llm_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase in your style. Do not answer.",
    )

    assert result.draft_response == "tối nay ăn j đây"

    # Verify rewrite_prompt was used, not system_prompt
    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert any("Rephrase" in m["content"] for m in system_msgs)


def test_responder_includes_critic_feedback_on_retry():
    """On retry, critic feedback is included in the prompt."""
    client = _make_mock_llm_client("tối nay ăn j đây")
    state = PipelineState(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        intent="rewrite_casual",
        tone="casual",
        resolved_content="Tối nay mình ăn gì nhỉ",
        tone_chunks=[],
        content_chunks=[],
        critic_feedback="You answered instead of rephrasing. Rephrase the question.",
        retry_count=1,
    )

    result = responder_agent(
        state,
        llm_client=client,
        llm_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase in your style.",
    )

    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    all_content = " ".join(m["content"] for m in messages)
    assert "answered instead of rephrasing" in all_content

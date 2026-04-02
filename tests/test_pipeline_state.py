"""Tests for PipelineState dataclass."""

from app.pipeline.state import PipelineState


def test_pipeline_state_defaults():
    """PipelineState initializes with sensible defaults."""
    state = PipelineState(raw_input="hello", mode="answer")

    assert state.raw_input == "hello"
    assert state.mode == "answer"
    assert state.intent is None
    assert state.tone is None
    assert state.needs_context is False
    assert state.context_source is None
    assert state.context_url is None
    assert state.resolved_content is None
    assert state.tone_chunks == []
    assert state.content_chunks == []
    assert state.draft_response is None
    assert state.approved is False
    assert state.critic_feedback is None
    assert state.retry_count == 0


def test_pipeline_state_rewrite_mode():
    """PipelineState works for rewrite mode."""
    state = PipelineState(raw_input="rewrite this article", mode="rewrite")
    assert state.mode == "rewrite"


def test_pipeline_state_is_mutable():
    """PipelineState fields can be updated (needed for LangGraph node updates)."""
    state = PipelineState(raw_input="test", mode="answer")
    state.intent = "casual_chat"
    state.tone = "casual_banter"
    state.resolved_content = "test"
    state.approved = True
    assert state.intent == "casual_chat"
    assert state.approved is True

# tests/test_agent_context.py

from unittest.mock import MagicMock, patch
from app.pipeline.state import PipelineState
from app.pipeline.agents.context import context_agent


def test_context_agent_fetches_url():
    """Context agent fetches URL content and sets resolved_content."""
    state = PipelineState(
        raw_input="rewrite this https://example.com/article",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="https://example.com/article",
    )

    with patch("app.pipeline.agents.context.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>Article content here about technology.</p></body></html>"
        mock_httpx.get.return_value = mock_response

        # Also mock the intent re-classification LLM call
        llm_client = MagicMock()
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = '{"intent": "rewrite_article", "tone": "informational"}'
        llm_client.chat.completions.create.return_value = llm_resp

        result = context_agent(state, llm_client=llm_client, llm_model="test")

    assert result.resolved_content is not None
    assert len(result.resolved_content) > 0
    assert result.tone == "informational"


def test_context_agent_skips_when_not_needed():
    """Context agent is a no-op when needs_context is False."""
    state = PipelineState(
        raw_input="hello",
        mode="answer",
        needs_context=False,
        resolved_content="hello",
    )

    result = context_agent(state)

    assert result.resolved_content == "hello"


def test_context_agent_handles_fetch_failure():
    """Context agent falls back to raw_input on fetch failure."""
    state = PipelineState(
        raw_input="rewrite this https://example.com/broken",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="https://example.com/broken",
    )

    with patch("app.pipeline.agents.context.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("Connection failed")

        result = context_agent(state)

    # Falls back to raw_input minus the URL
    assert result.resolved_content is not None


def test_context_agent_handles_non_200_status():
    """Context agent falls back to raw_input on non-200 HTTP status."""
    state = PipelineState(
        raw_input="rewrite this https://example.com/gone",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="https://example.com/gone",
    )

    with patch("app.pipeline.agents.context.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_httpx.get.return_value = mock_response

        result = context_agent(state)

    assert result.resolved_content == state.raw_input

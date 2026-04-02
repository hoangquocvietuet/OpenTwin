# tests/test_agent_context.py

import pytest
from unittest.mock import MagicMock, patch
from app.pipeline.state import PipelineState
from app.pipeline.agents.context import context_agent, _is_safe_url


def test_context_agent_fetches_url():
    """Context agent fetches URL content and sets resolved_content."""
    state = PipelineState(
        raw_input="rewrite this https://example.com/article",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="https://example.com/article",
    )

    with patch("app.pipeline.agents.context.httpx") as mock_httpx, \
         patch("app.pipeline.agents.context.socket") as mock_socket:
        # Mock DNS resolution to return a public IP
        mock_socket.AF_UNSPEC = 0
        mock_socket.SOCK_STREAM = 1
        mock_socket.gaierror = OSError
        mock_socket.getaddrinfo.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/article"
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

    with patch("app.pipeline.agents.context.httpx") as mock_httpx, \
         patch("app.pipeline.agents.context.socket") as mock_socket:
        mock_socket.AF_UNSPEC = 0
        mock_socket.SOCK_STREAM = 1
        mock_socket.gaierror = OSError
        mock_socket.getaddrinfo.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        mock_httpx.get.side_effect = Exception("Connection failed")

        result = context_agent(state)

    # Falls back to raw_input minus the URL
    assert result.resolved_content is not None


# --- SSRF protection tests ---

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/article", True),
    ("http://example.com/page", True),
    ("ftp://example.com/file", False),
    ("file:///etc/passwd", False),
    ("gopher://evil.com", False),
    ("", False),
    ("https://", False),
    ("https://localhost/admin", False),
    ("https://0.0.0.0/internal", False),
    ("https://127.0.0.1/secret", False),
    ("https://10.0.0.1/internal", False),
    ("https://192.168.1.1/admin", False),
    ("https://172.16.0.1/private", False),
    ("https://169.254.169.254/metadata", False),  # cloud metadata endpoint
    ("https://93.184.216.34/page", True),  # public IP
])
def test_is_safe_url(url, expected):
    """_is_safe_url blocks private IPs, localhost, and non-http schemes."""
    # Mock DNS resolution so tests don't depend on real DNS
    with patch("app.pipeline.agents.context.socket") as mock_socket:
        mock_socket.AF_UNSPEC = 0
        mock_socket.SOCK_STREAM = 1
        mock_socket.gaierror = OSError
        # Return a public IP for domain lookups
        mock_socket.getaddrinfo.return_value = [
            (2, 1, 0, "", ("93.184.216.34", 0)),
        ]
        assert _is_safe_url(url) is expected


def test_is_safe_url_blocks_dns_rebinding():
    """_is_safe_url catches domains that resolve to private IPs (DNS rebinding)."""
    with patch("app.pipeline.agents.context.socket") as mock_socket:
        mock_socket.AF_UNSPEC = 0
        mock_socket.SOCK_STREAM = 1
        mock_socket.gaierror = OSError
        # evil.com resolves to 127.0.0.1
        mock_socket.getaddrinfo.return_value = [
            (2, 1, 0, "", ("127.0.0.1", 0)),
        ]
        assert _is_safe_url("https://evil.com/steal") is False


def test_context_agent_blocks_ssrf():
    """Context agent rejects internal URLs and falls back to raw_input."""
    state = PipelineState(
        raw_input="rewrite this http://169.254.169.254/latest/meta-data",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="http://169.254.169.254/latest/meta-data",
    )

    result = context_agent(state)

    # Should fall back to raw_input without making any HTTP request
    assert result.resolved_content == state.raw_input


def test_responder_hides_error_details():
    """Responder returns generic error, not raw exception details."""
    from app.pipeline.agents.responder import responder_agent

    llm_client = MagicMock()
    llm_client.chat.completions.create.side_effect = RuntimeError("secret internal error: API key abc123")

    state = PipelineState(
        raw_input="hello",
        mode="answer",
        resolved_content="hello",
    )

    result = responder_agent(state, llm_client=llm_client, llm_model="test", system_prompt="You are a twin.")

    assert "secret" not in result.draft_response
    assert "abc123" not in result.draft_response
    assert "try again" in result.draft_response.lower()

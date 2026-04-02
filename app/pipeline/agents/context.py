"""Context Agent — fetches external content when needed.

Conditional agent that only runs when the Intent Agent sets needs_context=True.
Fetches URLs, then re-classifies the resolved content's tone.
"""

import ipaddress
import json
import re
from urllib.parse import urlparse

import httpx

from app.pipeline.state import PipelineState

_HTML_TAG_RE = re.compile(r'<[^>]+>')
_WHITESPACE_RE = re.compile(r'\s+')


def _extract_text_from_html(html: str) -> str:
    """Naive HTML to text extraction."""
    # Remove script and style blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = _HTML_TAG_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def _reclassify_tone(content: str, mode: str, llm_client, llm_model: str) -> tuple[str, str]:
    """Re-run intent/tone classification on fetched content."""
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": (
                    "Classify this text. Return JSON: "
                    '{"intent": "...", "tone": "..."} '
                    "where intent is one of: rewrite_article, rewrite_casual, rewrite_announcement, "
                    "information_sharing, other; and tone is one of: casual_banter, casual, formal_news, "
                    "formal, informational, serious, sarcastic, emotional, neutral."
                )},
                {"role": "user", "content": f"Mode: {mode}\nText: {content[:2000]}"},
            ],
            timeout=15,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        return parsed.get("intent", "general"), parsed.get("tone", "neutral")
    except Exception:
        return "general", "neutral"


def _is_safe_url(url: str) -> bool:
    """Check that a URL is safe to fetch (no SSRF to internal networks)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Block localhost and loopback
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False
        # Block private/reserved IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            pass  # hostname is a domain name, not an IP — that's fine
        return True
    except Exception:
        return False


def context_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
) -> PipelineState:
    """Fetch external context if needed, then re-classify tone.

    Only operates when state.needs_context is True.
    """
    if not state.needs_context:
        return state

    if state.context_source == "url" and state.context_url:
        if not _is_safe_url(state.context_url):
            state.resolved_content = state.raw_input
            return state
        try:
            resp = httpx.get(state.context_url, timeout=15, follow_redirects=False)
            if resp.status_code == 200:
                text = _extract_text_from_html(resp.text)
                if text:
                    state.resolved_content = text[:10000]  # cap at 10k chars
                else:
                    state.resolved_content = state.raw_input
            else:
                state.resolved_content = state.raw_input
        except Exception:
            # Fall back to raw input minus URL
            url_removed = state.raw_input.replace(state.context_url, "").strip()
            state.resolved_content = url_removed or state.raw_input
    else:
        # clipboard or other — for now fall back to raw_input
        state.resolved_content = state.raw_input

    # Re-classify tone on the actual content
    if llm_client and state.resolved_content:
        intent, tone = _reclassify_tone(
            state.resolved_content, state.mode, llm_client, llm_model or "llama3.1:8b"
        )
        state.intent = intent
        state.tone = tone

    return state

"""Context Agent — fetches external content when needed.

Conditional agent that only runs when the Intent Agent sets needs_context=True.
Fetches URLs, then re-classifies the resolved content's tone.
"""

import ipaddress
import json
import re
import socket
from urllib.parse import urlparse

import httpx

from app.pipeline.state import PipelineState

_ALLOWED_SCHEMES = {"http", "https"}


def _is_private_ip(addr_str: str) -> bool:
    """Check if an IP address string is private/loopback/link-local/reserved."""
    try:
        addr = ipaddress.ip_address(addr_str)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        return False


def _is_safe_url(url: str) -> bool:
    """Validate URL to prevent SSRF attacks.

    Blocks private/loopback/link-local IPs, non-http(s) schemes, and
    resolves hostnames to catch DNS rebinding to internal addresses.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Block obvious localhost aliases
    if hostname in ("localhost", "0.0.0.0"):
        return False

    # Check if hostname is a literal IP
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return False
    except ValueError:
        # hostname is a domain name — resolve it to check the actual IP
        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for family, _type, _proto, _canonname, sockaddr in resolved:
                ip_str = sockaddr[0]
                if _is_private_ip(ip_str):
                    return False
        except socket.gaierror:
            return False

    return True

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
            resp = httpx.get(state.context_url, timeout=15, follow_redirects=True, max_redirects=5)
            # Validate final URL after redirects to prevent redirect-based SSRF
            final_url = str(resp.url)
            if final_url != state.context_url and not _is_safe_url(final_url):
                state.resolved_content = state.raw_input
                return state
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

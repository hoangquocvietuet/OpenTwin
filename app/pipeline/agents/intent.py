"""Intent Agent — classifies intent, tone, and detects context needs.

First agent in the pipeline. Determines how to route the message
and what retrieval strategy to use.
"""

import json
import re

from app.pipeline.state import PipelineState

_URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

_SYSTEM_PROMPT = """You classify chat messages. Given a message and its mode (answer or rewrite), return a JSON object:

- "intent": one of: casual_chat, question, greeting, banter, rewrite_article, rewrite_casual, rewrite_announcement, information_sharing, emotional, other, general
- "tone": one of: casual_banter, casual, playful, friendly, relaxed, formal_news, formal, informational, serious, sarcastic, humorous, emotional, vulnerable, personal, angry, confrontational, frustrated, neutral

Consider the message length, content, and mode. Return ONLY valid JSON."""


def intent_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
) -> PipelineState:
    """Classify the input message's intent and tone.

    Also detects whether external context (URL, clipboard) needs to be fetched.
    """
    raw = state.raw_input

    # Detect URL in input
    url_match = _URL_PATTERN.search(raw)

    # Short message with URL → needs context
    non_url_text = _URL_PATTERN.sub("", raw).strip()
    if url_match and len(non_url_text) < 100:
        state.needs_context = True
        state.context_source = "url"
        state.context_url = url_match.group(0)
    else:
        state.needs_context = False
        state.resolved_content = raw

    # Classify intent and tone via LLM
    if llm_client:
        try:
            response = llm_client.chat.completions.create(
                model=llm_model or "llama3.1:8b",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Mode: {state.mode}\nMessage: {raw[:2000]}"},
                ],
                timeout=15,
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)
            state.intent = parsed.get("intent", "general")
            state.tone = parsed.get("tone", "neutral")
        except Exception:
            state.intent = "general"
            state.tone = "neutral"
    else:
        state.intent = "general"
        state.tone = "neutral"

    return state

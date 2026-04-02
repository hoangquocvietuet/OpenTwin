"""Context analyzer — LLM-based chunk summarization.

Generates a context summary describing what's happening in a chunk,
using neighboring chunks for broader context awareness.
"""

import json

from app.analyzers.registry import AnalyzerInput


def _format_messages(messages: list[dict], max_messages: int = 15) -> str:
    """Format messages for LLM prompt."""
    lines = []
    for m in messages[:max_messages]:
        lines.append(f"{m.get('author', '?')}: {m.get('text', '')}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """You analyze chat conversations. Given a chunk of messages (and optionally surrounding context), return a JSON object with:

- "context_summary": 1-2 sentence description of what's happening (e.g. "friends planning dinner, relaxed tone")
- "interaction_type": one of: greeting, planning, banter, argument, support, information_sharing, story_telling, venting, flirting, business, other
- "relationship": one of: close_friends, acquaintances, romantic, family, colleagues, strangers, other

Return ONLY valid JSON, no markdown, no explanation."""


def analyze_context(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze chunk context using LLM."""
    defaults = {
        "context_summary": "general conversation",
        "interaction_type": "other",
        "relationship": "other",
    }

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    raw = ""
    parts = []
    if input.prev_chunk and input.prev_chunk.get("messages"):
        parts.append(f"[PREVIOUS CONTEXT]\n{_format_messages(input.prev_chunk['messages'], max_messages=5)}")
    parts.append(f"[CURRENT CHUNK]\n{_format_messages(input.chunk['messages'])}")
    if input.next_chunk and input.next_chunk.get("messages"):
        parts.append(f"[NEXT CONTEXT]\n{_format_messages(input.next_chunk['messages'], max_messages=5)}")

    user_prompt = "\n\n".join(parts)

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            timeout=30,
        )
        raw = response.choices[0].message.content or ""

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(cleaned)
        return {
            "context_summary": parsed.get("context_summary", defaults["context_summary"]),
            "interaction_type": parsed.get("interaction_type", defaults["interaction_type"]),
            "relationship": parsed.get("relationship", defaults["relationship"]),
        }
    except (json.JSONDecodeError, Exception):
        if raw and len(raw) < 200:
            return {**defaults, "context_summary": raw.strip()}
        return defaults

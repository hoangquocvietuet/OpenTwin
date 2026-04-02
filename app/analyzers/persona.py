"""Persona analyzer — LLM-based role and register classification.

Classifies how the twin behaves in this chunk: their role in the conversation,
their register (formality level with this person), and relationship type.
"""

import json

from app.analyzers.registry import AnalyzerInput

_SYSTEM_PROMPT = """Analyze the twin's persona in this conversation. The twin's name is: {twin_name}

Return a JSON object with:
- "twin_role": how the twin acts — one of: initiator, responder, leader, follower, mediator, provocateur, supporter, observer, participant
- "register": the twin's language register here — one of: informal_close, informal_casual, semi_formal, formal, code_switching, playful_vulgar, respectful_elder, unknown
- "relationship_to_others": one of: friend_banter, close_friend, acquaintance, romantic, family_casual, family_formal, colleague, mentor, mentee, stranger, other

Consider the context summary if provided. Return ONLY valid JSON."""


def analyze_persona(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze twin's persona in this chunk using LLM."""
    defaults = {"twin_role": "participant", "register": "unknown", "relationship_to_others": "other"}

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    messages_text = "\n".join(
        f"{m.get('author', '?')}: {m.get('text', '')}"
        for m in input.chunk["messages"][:15]
    )

    context_summary = input.chunk.get("metadata", {}).get("context_summary", "")
    context_line = f"\n\nContext: {context_summary}" if context_summary else ""

    system_prompt = _SYSTEM_PROMPT.replace("{twin_name}", twin_name)

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{messages_text}{context_line}"},
            ],
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "twin_role": parsed.get("twin_role", defaults["twin_role"]),
            "register": parsed.get("register", defaults["register"]),
            "relationship_to_others": parsed.get("relationship_to_others", defaults["relationship_to_others"]),
        }
    except (json.JSONDecodeError, ValueError, Exception):
        return defaults

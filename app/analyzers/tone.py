"""Tone analyzer — LLM-based tone/formality classification.

Classifies the overall tone, formality level, and energy of a chunk.
Uses context_summary from the context analyzer if available.
"""

import json

from app.analyzers.registry import AnalyzerInput

_SYSTEM_PROMPT = """Analyze the tone of this chat conversation chunk. Return a JSON object with:

- "tone": one of: casual_banter, casual, playful, friendly, formal, formal_news, informational, serious, sarcastic, humorous, emotional, angry, confrontational, frustrated, vulnerable, personal, relaxed, neutral
- "formality": float 0.0 (very informal/slang) to 1.0 (very formal/proper)
- "energy": one of: high, medium, low, relaxed

Consider the context summary if provided. Return ONLY valid JSON."""


def analyze_tone(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze chunk tone using LLM."""
    defaults = {"tone": "neutral", "formality": 0.5, "energy": "medium"}

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    messages_text = "\n".join(
        f"{m.get('author', '?')}: {m.get('text', '')}"
        for m in input.chunk["messages"][:15]
    )

    context_summary = input.chunk.get("metadata", {}).get("context_summary", "")
    context_line = f"\n\nContext: {context_summary}" if context_summary else ""

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"{messages_text}{context_line}"},
            ],
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "tone": parsed.get("tone", defaults["tone"]),
            "formality": float(parsed.get("formality", defaults["formality"])),
            "energy": parsed.get("energy", defaults["energy"]),
        }
    except (json.JSONDecodeError, ValueError, Exception):
        return defaults

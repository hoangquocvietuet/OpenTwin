"""Emotion analyzer — LLM-based emotion/sentiment/conflict detection.

Classifies the emotional state, sentiment, and whether conflict or sarcasm
is present in a chunk.
"""

import json

from app.analyzers.registry import AnalyzerInput

_SYSTEM_PROMPT = """Analyze the emotional content of this chat conversation chunk. Return a JSON object with:

- "emotion": primary emotion — one of: happy, playful, excited, neutral, tired, sad, frustrated, angry, anxious, vulnerable, nostalgic, sarcastic, bored
- "sentiment": float from -1.0 (very negative) to 1.0 (very positive), 0.0 is neutral
- "conflict": boolean — is there a disagreement, argument, or tension between participants?
- "sarcasm": boolean — is sarcasm or irony being used?

Consider the context summary if provided. Return ONLY valid JSON."""


def analyze_emotion(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze chunk emotion using LLM."""
    defaults = {"emotion": "neutral", "sentiment": 0.0, "conflict": False, "sarcasm": False}

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
            "emotion": parsed.get("emotion", defaults["emotion"]),
            "sentiment": float(parsed.get("sentiment", defaults["sentiment"])),
            "conflict": bool(parsed.get("conflict", defaults["conflict"])),
            "sarcasm": bool(parsed.get("sarcasm", defaults["sarcasm"])),
        }
    except (json.JSONDecodeError, ValueError, Exception):
        return defaults

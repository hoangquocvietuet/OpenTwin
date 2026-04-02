"""LLM-based conversation boundary detection.

Scans a list of messages and identifies natural break points where
the topic, mood, or context shifts significantly.
"""

import json

_SYSTEM_PROMPT = """You analyze chat conversations to find natural break points.

Given a list of messages, identify indices where a NEW conversation segment starts. A new segment means:
- Topic changes (food → work, personal → news)
- Mood shifts (playful → serious, calm → angry)
- New participant enters and changes the dynamic
- Time gap + clear context shift (time gap alone is NOT enough if topic continues)

Return a JSON object: {"boundaries": [list of message indices where new segments start]}

Example: messages 0-4 about food, messages 5-9 about work → {"boundaries": [5]}

If the entire conversation is one coherent segment, return {"boundaries": []}
Return ONLY valid JSON."""


def _format_messages_for_boundary(messages: list[dict]) -> str:
    """Format messages with indices for boundary detection."""
    lines = []
    for i, m in enumerate(messages):
        ts = m.get("timestamp", "")
        author = m.get("author", "?")
        text = m.get("text", "")
        lines.append(f"[{i}] {ts} {author}: {text}")
    return "\n".join(lines)


def detect_boundaries(
    messages: list[dict],
    llm_client=None,
    llm_model: str | None = None,
    window_size: int = 40,
) -> list[int]:
    """Detect conversation boundaries in a list of messages.

    Uses a sliding window approach for long conversations.

    Args:
        messages: List of message dicts with author, text, timestamp
        llm_client: OpenAI-compatible client
        llm_model: Model name
        window_size: Max messages to send to LLM at once

    Returns:
        Sorted list of boundary indices (where new segments start).
    """
    if not llm_client or len(messages) < 3:
        return []

    all_boundaries: set[int] = set()

    step = window_size - 5
    for start in range(0, len(messages), max(step, 1)):
        window = messages[start:start + window_size]
        if len(window) < 3:
            break

        formatted = _format_messages_for_boundary(window)

        try:
            response = llm_client.chat.completions.create(
                model=llm_model or "llama3.1:8b",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": formatted},
                ],
                timeout=30,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            for idx in parsed.get("boundaries", []):
                global_idx = start + int(idx)
                if 0 < global_idx < len(messages):
                    all_boundaries.add(global_idx)
        except (json.JSONDecodeError, ValueError, Exception):
            continue

        if start + window_size >= len(messages):
            break

    return sorted(all_boundaries)

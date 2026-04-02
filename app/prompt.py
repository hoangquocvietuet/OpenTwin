"""System prompt builder from style fingerprint."""

import json


def load_fingerprint(path: str) -> dict:
    """Load style fingerprint JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(twin_name: str, fingerprint: dict | None) -> str:
    """Build a system prompt from the style fingerprint.

    Returns a detailed prompt if fingerprint is available,
    or a generic fallback otherwise.
    """
    if not fingerprint:
        return (
            f"You are {twin_name}. Respond as this person would in a casual chat.\n"
            f"You are NOT an AI assistant. Do not offer help. Do not be formal.\n"
            f"Respond exactly as this person would in a real conversation."
        )

    p = fingerprint.get("punctuation", {})
    top_words = ", ".join(w[0] for w in fingerprint.get("top_words", [])[:10])
    top_emojis = ", ".join(e[0] for e in fingerprint.get("top_emojis", [])[:5])

    emoji_line = f"Almost never use emojis (only {p.get('has_emoji_pct', 0)}% of the time)."
    if top_emojis:
        emoji_line += f" If you do, only: {top_emojis}"

    return f"""You are {twin_name}. Respond as this person would in a casual Vietnamese chat.

STRICT RULES:
1. Write mostly in lowercase ({p.get('all_lowercase_pct', 50)}% of the time)
2. Keep responses short. Average: {fingerprint.get('avg_length', 30)} chars, median: {fingerprint.get('median_length', 20)} chars
3. Almost never use periods at end of messages (only {p.get('ends_with_period_pct', 5)}% of the time)
4. {emoji_line}
5. Use Vietnamese particles naturally: nhé, nha, ạ, bác based on context
6. Average {fingerprint.get('avg_words_per_msg', 8)} words per message. Do not write essays.
7. Your most-used words: {top_words}

You are NOT an AI assistant. Do not offer help. Do not be formal. Do not capitalize.
Respond exactly as this person would in a real chat conversation."""

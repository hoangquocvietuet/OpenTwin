"""Stats analyzer — heuristic, no LLM required.

Extracts message-level statistics from a chunk: counts, ratios,
emoji usage, question detection, language detection.
"""

import re

# Vietnamese-specific characters beyond ASCII
_VIETNAMESE_PATTERN = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', re.IGNORECASE)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U0001f900-\U0001f9FF"
    "]+",
    flags=re.UNICODE,
)
_QUESTION_PATTERN = re.compile(r'[?？]|(\b(gì|j|sao|nào|ko|không|chưa|hả|đâu|bao giờ|mấy|ai)\b)', re.IGNORECASE)


def _detect_language(texts: list[str]) -> str:
    """Detect primary language from a list of texts. Returns 'vi', 'en', or 'mixed'."""
    if not texts:
        return "vi"

    vi_count = 0
    en_count = 0
    for text in texts:
        if _VIETNAMESE_PATTERN.search(text):
            vi_count += 1
        elif re.search(r'[a-zA-Z]{3,}', text):
            en_count += 1

    total = vi_count + en_count
    if total == 0:
        return "vi"
    if vi_count / total > 0.7:
        return "vi"
    if en_count / total > 0.7:
        return "en"
    return "mixed"


def analyze_stats(chunk: dict, twin_name: str) -> dict:
    """Extract heuristic statistics from a chunk. No LLM required.

    Args:
        chunk: Dict with "messages" list (each has "author", "text", "timestamp")
        twin_name: Name of the twin to calculate twin-specific stats

    Returns:
        Dict with stats metadata fields.
    """
    messages = chunk.get("messages", [])
    if not messages:
        return {
            "msg_count": 0,
            "twin_msg_count": 0,
            "twin_msg_ratio": 0.0,
            "avg_msg_len": 0.0,
            "twin_avg_msg_len": 0.0,
            "emoji_count": 0,
            "question_ratio": 0.0,
            "language": "vi",
        }

    texts = [m.get("text") or "" for m in messages]
    twin_texts = [m.get("text") or "" for m in messages if m.get("author") == twin_name]

    all_lengths = [len(t) for t in texts]
    twin_lengths = [len(t) for t in twin_texts]

    emoji_count = sum(len(_EMOJI_PATTERN.findall(t)) for t in texts)
    question_count = sum(1 for t in texts if _QUESTION_PATTERN.search(t))

    return {
        "msg_count": len(messages),
        "twin_msg_count": len(twin_texts),
        "twin_msg_ratio": round(len(twin_texts) / len(messages), 3) if messages else 0.0,
        "avg_msg_len": round(sum(all_lengths) / len(all_lengths), 1) if all_lengths else 0.0,
        "twin_avg_msg_len": round(sum(twin_lengths) / len(twin_lengths), 1) if twin_lengths else 0.0,
        "emoji_count": emoji_count,
        "question_ratio": round(question_count / len(messages), 3) if messages else 0.0,
        "language": _detect_language(texts),
    }

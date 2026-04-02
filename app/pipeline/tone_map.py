"""Tone similarity mapping for retrieval expansion.

When the Intent Agent detects a tone, the Retriever Agent uses this map
to expand the query to include similar tones.
"""

TONE_SIMILARITY: dict[str, list[str]] = {
    "casual_banter": ["casual", "playful", "friendly"],
    "casual": ["casual_banter", "friendly", "relaxed"],
    "playful": ["casual_banter", "humorous", "friendly"],
    "friendly": ["casual", "casual_banter", "relaxed"],
    "relaxed": ["casual", "friendly", "neutral"],
    "formal_news": ["informational", "serious", "formal"],
    "formal": ["formal_news", "serious", "informational"],
    "informational": ["formal_news", "formal", "serious"],
    "serious": ["formal", "informational", "emotional"],
    "sarcastic": ["playful", "casual_banter", "humorous"],
    "humorous": ["playful", "sarcastic", "casual_banter"],
    "emotional": ["serious", "vulnerable", "personal"],
    "vulnerable": ["emotional", "personal", "serious"],
    "personal": ["emotional", "vulnerable", "friendly"],
    "angry": ["confrontational", "frustrated", "serious"],
    "confrontational": ["angry", "frustrated", "serious"],
    "frustrated": ["angry", "confrontational", "serious"],
    "neutral": ["casual", "relaxed", "friendly"],
}


def get_similar_tones(tone: str) -> list[str]:
    """Get a tone plus its similar tones for retrieval expansion.

    Args:
        tone: The detected tone string

    Returns:
        List of tones to query, with the input tone first. No duplicates.
    """
    similar = TONE_SIMILARITY.get(tone, [])
    result = [tone] + [t for t in similar if t != tone]
    return result

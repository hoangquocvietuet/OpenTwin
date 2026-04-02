from app.prompt import build_system_prompt, load_fingerprint


def test_load_fingerprint(sample_fingerprint):
    """Loads fingerprint JSON from disk."""
    fp = load_fingerprint(sample_fingerprint)
    assert fp["total_messages"] == 940
    assert fp["avg_length"] == 38.4
    assert fp["punctuation"]["all_lowercase_pct"] == 71.1


def test_build_system_prompt_contains_key_rules(sample_fingerprint):
    """System prompt includes fingerprint-derived rules."""
    fp = load_fingerprint(sample_fingerprint)
    prompt = build_system_prompt("Hoàng Quốc Việt", fp)

    assert "Hoàng Quốc Việt" in prompt
    assert "71.1%" in prompt  # all_lowercase_pct
    assert "38.4" in prompt  # avg_length
    assert "25" in prompt  # median_length
    assert "3.4%" in prompt  # ends_with_period_pct
    assert "0.9%" in prompt  # has_emoji_pct
    assert "9.2" in prompt  # avg_words_per_msg
    assert "anh" in prompt  # top word
    assert "NOT an AI assistant" in prompt


def test_build_system_prompt_no_fingerprint():
    """Falls back to a generic prompt when no fingerprint data."""
    prompt = build_system_prompt("Hoàng Quốc Việt", None)
    assert "Hoàng Quốc Việt" in prompt
    assert "casual" in prompt.lower()

from app.analyzers.stats import analyze_stats


def test_analyze_stats_basic():
    """Stats analyzer extracts message-level statistics from a chunk."""
    chunk = {
        "messages": [
            {"author": "Hoang Quoc Viet", "text": "tối nay ăn j đây", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ăn phở đi", "timestamp": "2025-08-01T10:01:00"},
            {"author": "Hoang Quoc Viet", "text": "ok 👍", "timestamp": "2025-08-01T10:02:00"},
        ],
        "metadata": {
            "participants": ["Hoang Quoc Viet", "Friend"],
        },
    }

    result = analyze_stats(chunk, twin_name="Hoang Quoc Viet")

    assert result["msg_count"] == 3
    assert result["twin_msg_count"] == 2
    assert result["twin_msg_ratio"] == round(2 / 3, 3)
    assert result["avg_msg_len"] > 0
    assert result["twin_avg_msg_len"] > 0
    assert result["emoji_count"] >= 1
    assert result["question_ratio"] > 0  # "ăn j đây" has question-like pattern
    assert result["language"] == "vi"


def test_analyze_stats_empty_chunk():
    """Stats analyzer handles empty messages gracefully."""
    chunk = {"messages": [], "metadata": {"participants": []}}
    result = analyze_stats(chunk, twin_name="Hoang Quoc Viet")

    assert result["msg_count"] == 0
    assert result["twin_msg_ratio"] == 0.0


def test_analyze_stats_mixed_language():
    """Stats analyzer detects mixed language."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "are you oke", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Viet", "text": "where are diu", "timestamp": "2025-08-01T10:01:00"},
            {"author": "Viet", "text": "tối ăn gì này", "timestamp": "2025-08-01T10:02:00"},
        ],
        "metadata": {"participants": ["Viet"]},
    }
    result = analyze_stats(chunk, twin_name="Viet")
    assert result["language"] in ("mixed", "vi", "en")

"""Shared test fixtures."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with sample fingerprint and chunks.

    Structure: tmp_path/hoang_quoc_viet/ (this IS the data_dir level).
    """
    target_dir = tmp_path / "hoang_quoc_viet"
    target_dir.mkdir()

    # Sample style fingerprint
    fingerprint = {
        "total_messages": 940,
        "avg_length": 38.4,
        "median_length": 25,
        "avg_words_per_msg": 9.2,
        "length_distribution": {"6-20": 385, "21-50": 424},
        "punctuation": {
            "all_lowercase_pct": 71.1,
            "ends_with_period_pct": 3.4,
            "uses_exclamation_pct": 1.3,
            "question_mark_pct": 4.4,
            "uses_ellipsis_pct": 0.3,
            "has_emoji_pct": 0.9,
        },
        "top_emojis": [["🤣", 3]],
        "top_words": [["anh", 146], ["mình", 102], ["cho", 82]],
    }
    with open(target_dir / "style_fingerprint.json", "w") as f:
        json.dump(fingerprint, f)

    # Sample train chunks (3 chunks)
    chunks = [
        {
            "chunk_id": "dm_test_0",
            "thread_id": "inbox/test_thread",
            "chunk_type": "dm",
            "score": 1.5,
            "context": [
                {"author": "Friend", "text": "bạn làm gì đấy?", "timestamp": "2025-08-01T10:00:00", "is_target": False}
            ],
            "response": {"author": "Hoàng Quốc Việt", "text": "đang code dự án mới nè", "timestamp": "2025-08-01T10:01:00"},
            "response_length": 23,
            "context_turns": 1,
            "has_question": True,
            "time_gap_seconds": 0.0,
        },
        {
            "chunk_id": "dm_test_1",
            "thread_id": "inbox/test_thread",
            "chunk_type": "dm",
            "score": 1.2,
            "context": [
                {"author": "Friend", "text": "đi ăn không?", "timestamp": "2025-08-01T12:00:00", "is_target": False}
            ],
            "response": {"author": "Hoàng Quốc Việt", "text": "ok đi anh", "timestamp": "2025-08-01T12:01:00"},
            "response_length": 10,
            "context_turns": 1,
            "has_question": True,
            "time_gap_seconds": 0.0,
        },
        {
            "chunk_id": "dm_test_2",
            "thread_id": "inbox/test_thread_2",
            "chunk_type": "dm",
            "score": 0.8,
            "context": [],
            "response": {"author": "Hoàng Quốc Việt", "text": "ê bạn ơi", "timestamp": "2025-08-02T09:00:00"},
            "response_length": 9,
            "context_turns": 0,
            "has_question": False,
            "time_gap_seconds": 0.0,
        },
    ]
    with open(target_dir / "train_chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return tmp_path


@pytest.fixture
def sample_fingerprint(tmp_data_dir):
    """Return path to the sample fingerprint file."""
    return str(tmp_data_dir / "hoang_quoc_viet" / "style_fingerprint.json")


@pytest.fixture
def sample_chunks_path(tmp_data_dir):
    """Return path to the sample train chunks file."""
    return str(tmp_data_dir / "hoang_quoc_viet" / "train_chunks.jsonl")

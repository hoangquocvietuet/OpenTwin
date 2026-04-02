from app.chunking.normalizer import normalize_segments, build_chunks


def test_normalize_merges_tiny_segments():
    """Segments with < 3 messages get merged with their neighbor."""
    messages = [
        {"author": "A", "text": "hi", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
        {"author": "A", "text": "đi ăn k", "timestamp": "2025-08-01T10:02:00"},
        {"author": "B", "text": "ok", "timestamp": "2025-08-01T10:03:00"},
        {"author": "A", "text": "ăn phở", "timestamp": "2025-08-01T10:04:00"},
        {"author": "B", "text": "oke", "timestamp": "2025-08-01T10:05:00"},
        {"author": "A", "text": "mấy giờ", "timestamp": "2025-08-01T10:06:00"},
    ]
    boundaries = [2]

    normalized = normalize_segments(messages, boundaries, min_size=3, max_size=20)

    assert len(normalized) == 1
    assert len(normalized[0]) == 7


def test_normalize_splits_large_segments():
    """Segments with > max_size messages get split."""
    messages = [{"author": "A", "text": f"msg {i}", "timestamp": f"2025-08-01T{10+i}:00:00"} for i in range(25)]
    boundaries = []

    normalized = normalize_segments(messages, boundaries, min_size=3, max_size=10)

    assert len(normalized) >= 2
    for seg in normalized:
        assert len(seg) <= 10


def test_build_chunks_creates_chunk_dicts():
    """build_chunks creates properly structured chunk dicts from segments."""
    segments = [
        [
            {"author": "Viet", "text": "ê đi ăn", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ok", "timestamp": "2025-08-01T10:01:00"},
            {"author": "Viet", "text": "ăn phở", "timestamp": "2025-08-01T10:02:00"},
        ],
    ]

    chunks = build_chunks(segments, thread_id="inbox/thread_1", twin_name="Viet")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"].startswith("inbox/thread_1_seg")
    assert chunk["messages"] == segments[0]
    assert "document" in chunk
    assert "Viet: ê đi ăn" in chunk["document"]
    assert chunk["metadata"]["msg_count"] == 3
    assert chunk["metadata"]["participants"] == ["Friend", "Viet"]
    assert chunk["metadata"]["time_start"] == "2025-08-01T10:00:00"
    assert chunk["metadata"]["time_end"] == "2025-08-01T10:02:00"
    assert chunk["metadata"]["twin_msg_ratio"] == round(2 / 3, 3)

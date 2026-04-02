# tests/test_rechunk.py

import json
import os
from unittest.mock import MagicMock, patch

from app.rechunk import load_raw_messages_from_sources, rechunk_twin


def test_load_raw_messages_from_sources(tmp_path):
    """Loads raw messages from source train_chunks.jsonl files."""
    twin_dir = tmp_path / "data" / "hoang_quoc_viet"
    source_dir = twin_dir / "sources" / "src_001"
    source_dir.mkdir(parents=True)

    chunks = [
        {
            "chunk_id": "dm_test_0",
            "thread_id": "inbox/thread_1",
            "context": [
                {"author": "Friend", "text": "hi", "timestamp": "2025-08-01T10:00:00", "is_target": False}
            ],
            "response": {"author": "Viet", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
        },
        {
            "chunk_id": "dm_test_1",
            "thread_id": "inbox/thread_1",
            "context": [],
            "response": {"author": "Viet", "text": "what's up", "timestamp": "2025-08-01T10:02:00"},
        },
    ]
    with open(source_dir / "train_chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    # Create sources.json
    sources = [{"id": "src_001", "enabled": True, "name": "test"}]
    with open(twin_dir / "sources.json", "w") as f:
        json.dump(sources, f)

    messages_by_thread = load_raw_messages_from_sources(
        data_dir=str(tmp_path / "data"),
        twin_slug="hoang_quoc_viet",
    )

    assert "inbox/thread_1" in messages_by_thread
    assert len(messages_by_thread["inbox/thread_1"]) >= 2


def test_rechunk_twin_produces_chunks(tmp_path):
    """rechunk_twin produces chunks from raw messages."""
    # Setup minimal raw data
    twin_dir = tmp_path / "data" / "test_twin"
    source_dir = twin_dir / "sources" / "src_001"
    source_dir.mkdir(parents=True)

    chunks = [
        {
            "chunk_id": f"dm_{i}",
            "thread_id": "inbox/thread_1",
            "context": [{"author": "Friend", "text": f"msg {i}", "timestamp": f"2025-08-01T{10+i}:00:00", "is_target": False}],
            "response": {"author": "Viet", "text": f"reply {i}", "timestamp": f"2025-08-01T{10+i}:01:00"},
        }
        for i in range(5)
    ]
    with open(source_dir / "train_chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    sources = [{"id": "src_001", "enabled": True, "name": "test"}]
    with open(twin_dir / "sources.json", "w") as f:
        json.dump(sources, f)

    # Mock LLM for boundary detection (no boundaries = 1 chunk)
    llm_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"boundaries": []}'
    llm_client.chat.completions.create.return_value = mock_resp

    result_chunks = rechunk_twin(
        data_dir=str(tmp_path / "data"),
        twin_slug="test_twin",
        twin_name="Viet",
        llm_client=llm_client,
        llm_model="test",
    )

    assert len(result_chunks) >= 1
    assert all("chunk_id" in c for c in result_chunks)
    assert all("document" in c for c in result_chunks)
    assert all("metadata" in c for c in result_chunks)

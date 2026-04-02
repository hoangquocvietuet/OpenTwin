# tests/test_importer_chunking.py

import sys
from unittest.mock import MagicMock, patch

from app.importer import _enrich_chunks_with_analyzers


def test_enrich_returns_chunks_unchanged_when_no_analyzer_configured():
    """Without analyzer config, chunks are returned as-is."""
    chunks = [{"chunk_id": "c1", "document": "test", "metadata": {}}]
    result = _enrich_chunks_with_analyzers(chunks, twin_name="Viet")
    assert result == chunks


def test_enrich_returns_chunks_unchanged_when_no_model():
    """Without analyzer model, chunks are returned as-is."""
    chunks = [{"chunk_id": "c1", "document": "test", "metadata": {}}]
    result = _enrich_chunks_with_analyzers(
        chunks, twin_name="Viet",
        analyzer_base_url="http://localhost:11434/v1",
        analyzer_model=None,
    )
    assert result == chunks


def test_enrich_returns_chunks_unchanged_when_no_base_url():
    """Without analyzer base URL, chunks are returned as-is."""
    chunks = [{"chunk_id": "c1", "document": "test", "metadata": {}}]
    result = _enrich_chunks_with_analyzers(
        chunks, twin_name="Viet",
        analyzer_base_url=None,
        analyzer_model="llama3",
    )
    assert result == chunks


def test_enrich_calls_analyzers_when_configured():
    """With analyzer config, run_analyzers is called for each chunk."""
    chunks = [
        {"chunk_id": "c1", "document": "hello", "metadata": {}},
        {"chunk_id": "c2", "document": "world", "metadata": {}},
    ]

    # The function does lazy imports inside the body, so we patch sys.modules
    # to inject mock modules before the function's import statements execute.
    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI.return_value = MagicMock()

    mock_registry = MagicMock()
    mock_default_registry_mod = MagicMock()
    mock_default_registry_mod.create_default_registry.return_value = mock_registry

    mock_run_analyzers = MagicMock(
        return_value={"tone": "casual", "_analyzers_applied": {"tone_v1": 1}}
    )
    mock_registry_mod = MagicMock()
    mock_registry_mod.run_analyzers = mock_run_analyzers

    with patch.dict(sys.modules, {
        "openai": mock_openai_module,
        "app.analyzers.default_registry": mock_default_registry_mod,
        "app.analyzers.registry": mock_registry_mod,
    }):
        result = _enrich_chunks_with_analyzers(
            chunks, twin_name="Viet",
            analyzer_base_url="http://localhost:11434/v1",
            analyzer_model="test-model",
            analyzer_api_key="test-key",
        )

    assert mock_run_analyzers.call_count == 2
    # Verify metadata was updated for every chunk
    for chunk in result:
        assert "tone" in chunk["metadata"]
        assert chunk["metadata"]["tone"] == "casual"


def test_enrich_handles_import_error_gracefully():
    """If analyzer modules are missing, ImportError is caught and chunks returned as-is."""
    chunks = [{"chunk_id": "c1", "document": "test", "metadata": {}}]

    # Remove openai from sys.modules and make it un-importable
    original = sys.modules.pop("openai", None)
    try:
        with patch.dict(sys.modules, {"openai": None}):
            result = _enrich_chunks_with_analyzers(
                chunks, twin_name="Viet",
                analyzer_base_url="http://localhost:11434/v1",
                analyzer_model="test-model",
            )
        assert result == chunks
    finally:
        if original is not None:
            sys.modules["openai"] = original


def test_enrich_passes_correct_prev_next_context():
    """run_analyzers receives correct prev/next chunk references."""
    chunks = [
        {"chunk_id": "c1", "document": "first", "metadata": {}},
        {"chunk_id": "c2", "document": "second", "metadata": {}},
        {"chunk_id": "c3", "document": "third", "metadata": {}},
    ]

    mock_openai_module = MagicMock()
    mock_default_registry_mod = MagicMock()
    mock_default_registry_mod.create_default_registry.return_value = MagicMock()

    call_args_list = []

    def capturing_run_analyzers(registry, chunk, **kwargs):
        call_args_list.append({
            "chunk": chunk,
            "prev_chunk": kwargs.get("prev_chunk"),
            "next_chunk": kwargs.get("next_chunk"),
        })
        return {}

    mock_registry_mod = MagicMock()
    mock_registry_mod.run_analyzers.side_effect = capturing_run_analyzers

    with patch.dict(sys.modules, {
        "openai": mock_openai_module,
        "app.analyzers.default_registry": mock_default_registry_mod,
        "app.analyzers.registry": mock_registry_mod,
    }):
        _enrich_chunks_with_analyzers(
            chunks, twin_name="Viet",
            analyzer_base_url="http://localhost:11434/v1",
            analyzer_model="test-model",
        )

    assert len(call_args_list) == 3
    # First chunk: no prev, has next
    assert call_args_list[0]["prev_chunk"] is None
    assert call_args_list[0]["next_chunk"] is chunks[1]
    # Middle chunk: has both
    assert call_args_list[1]["prev_chunk"] is chunks[0]
    assert call_args_list[1]["next_chunk"] is chunks[2]
    # Last chunk: has prev, no next
    assert call_args_list[2]["prev_chunk"] is chunks[1]
    assert call_args_list[2]["next_chunk"] is None

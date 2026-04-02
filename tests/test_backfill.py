from unittest.mock import MagicMock, patch
from app.backfill import find_chunks_needing_backfill, backfill_collection


def test_find_chunks_needing_backfill():
    """Identifies chunks missing analyzers or with outdated versions."""
    collection = MagicMock()
    collection.count.return_value = 2
    collection.get.return_value = {
        "ids": ["chunk_0", "chunk_1"],
        "metadatas": [
            {"_analyzers_applied": '{"stats_v1": 1}'},
            {"_analyzers_applied": '{"stats_v1": 1, "tone_v1": 1}'},
        ],
    }

    from app.analyzers.registry import AnalyzerRegistry
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=lambda *a, **kw: {}, version=1, requires_llm=False, run_order=0)
    registry.register("tone_v1", fn=lambda *a, **kw: {}, version=1, requires_llm=True, run_order=1)

    needs_backfill = find_chunks_needing_backfill(collection, registry)

    assert "chunk_0" in needs_backfill
    assert "chunk_1" not in needs_backfill


def test_find_chunks_needing_backfill_outdated_version():
    """Chunks with outdated analyzer version need backfill."""
    collection = MagicMock()
    collection.count.return_value = 1
    collection.get.return_value = {
        "ids": ["chunk_0"],
        "metadatas": [
            {"_analyzers_applied": '{"stats_v1": 1}'},
        ],
    }

    from app.analyzers.registry import AnalyzerRegistry
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=lambda *a, **kw: {}, version=2, requires_llm=False, run_order=0)

    needs_backfill = find_chunks_needing_backfill(collection, registry)

    assert "chunk_0" in needs_backfill

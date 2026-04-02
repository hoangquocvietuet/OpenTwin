from unittest.mock import MagicMock
from app.pipeline.detect import has_enriched_metadata


def test_enriched_collection_detected():
    """Collection with tone metadata is detected as enriched."""
    collection = MagicMock()
    collection.count.return_value = 10
    collection.get.return_value = {
        "ids": ["chunk_0"],
        "metadatas": [{"tone": "casual", "_analyzers_applied": '{"tone_v1": 1}'}],
    }

    assert has_enriched_metadata(collection) is True


def test_unenriched_collection_detected():
    """Collection without tone metadata is detected as unenriched."""
    collection = MagicMock()
    collection.count.return_value = 10
    collection.get.return_value = {
        "ids": ["chunk_0"],
        "metadatas": [{"chunk_type": "dm", "score": 1.5}],
    }

    assert has_enriched_metadata(collection) is False


def test_empty_collection():
    """Empty collection returns False."""
    collection = MagicMock()
    collection.count.return_value = 0

    assert has_enriched_metadata(collection) is False

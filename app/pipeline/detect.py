"""Detect whether a ChromaDB collection has enriched metadata.

Used by chat_service to decide whether to use the new pipeline
or fall back to legacy code.
"""

import logging

logger = logging.getLogger(__name__)


def has_enriched_metadata(collection) -> bool:
    """Check if the collection has analyzer-enriched metadata.

    Samples one chunk and checks for the 'tone' field,
    which is only present after analyzer enrichment.
    Note: samples only one chunk — assumes the collection is
    uniformly enriched or unenriched (partial enrichment will
    give non-deterministic results).

    Args:
        collection: ChromaDB collection

    Returns:
        True if enriched metadata is present.
    """
    if collection.count() == 0:
        return False

    try:
        sample = collection.get(limit=1, include=["metadatas"])
        if not sample["metadatas"]:
            return False

        meta = sample["metadatas"][0]
        return "tone" in meta and "_analyzers_applied" in meta
    except Exception as exc:
        logger.warning("enriched metadata check failed: %s", exc)
        return False

"""ChromaDB query and context assembly for RAG."""

import chromadb


def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    min_score: float = 0.5,
    max_distance: float = 0.85,
) -> list[dict]:
    """Query ChromaDB collection and return ranked results.

    Fetches more candidates than needed, then filters and re-ranks by
    combining embedding similarity with chunk quality score.

    Returns list of dicts with: chunk_id, document, distance, metadata.
    Returns empty list if collection is empty.
    """
    if collection.count() == 0:
        return []

    # Fetch extra candidates to allow filtering
    fetch_n = min(n_results * 4, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        include=["documents", "distances", "metadatas"],
    )

    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "chunk_id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    # Filter out low-quality chunks
    filtered = [c for c in candidates if c["metadata"].get("score", 0) >= min_score]

    # If filtering removed everything, fall back to unfiltered
    if not filtered:
        filtered = candidates

    # Re-rank: combine embedding similarity with chunk quality score
    # DM chunks get a boost since they're more authentic conversations
    for c in filtered:
        sim = 1 - c["distance"]
        quality = c["metadata"].get("score", 0)
        dm_boost = 0.1 if c["metadata"].get("chunk_type") == "dm" else 0.0
        c["_rank_score"] = (sim * 0.6) + (quality / 2.0 * 0.3) + dm_boost

    filtered.sort(key=lambda c: c["_rank_score"], reverse=True)

    # Drop results with very low similarity.
    # We keep this a bit lenient because short/chatty queries can be noisy.
    distance_filtered = [c for c in filtered if c["distance"] < max_distance]
    if distance_filtered:
        filtered = distance_filtered
    # else: keep the best candidates rather than returning 0 chunks

    # Clean up internal key and return top N
    for c in filtered:
        del c["_rank_score"]

    return filtered[:n_results]


def format_few_shot_examples(retrieved: list[dict], max_examples: int = 3) -> str:
    """Format retrieved chunks as few-shot examples for the LLM prompt.

    Each example shows a conversation snippet the twin actually had.
    """
    if not retrieved:
        return ""

    examples = []
    for chunk in retrieved[:max_examples]:
        examples.append(chunk["document"])

    return "\n\n---\n\n".join(examples)

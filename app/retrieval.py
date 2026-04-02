"""ChromaDB query and context assembly for RAG."""

import chromadb


def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """Query ChromaDB collection and return ranked results.

    Returns list of dicts with: chunk_id, document, distance, metadata.
    Returns empty list if collection is empty.
    """
    if collection.count() == 0:
        return []

    # Don't request more results than documents exist
    actual_n = min(n_results, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=actual_n,
        include=["documents", "distances", "metadatas"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return chunks


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

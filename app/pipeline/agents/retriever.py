"""Retriever Agent — hybrid tone + content retrieval from ChromaDB.

Performs two queries:
1. Tone retrieval: find chunks matching the detected tone (style reference)
2. Content retrieval: find chunks matching the content (personal context)
"""

from app.pipeline.state import PipelineState
from app.pipeline.tone_map import get_similar_tones

# Content similarity threshold (cosine distance). Lower = more similar.
# Only keep content chunks with distance below this.
CONTENT_MAX_DISTANCE = 0.6

TONE_RESULTS_COUNT = 3
CONTENT_RESULTS_COUNT = 2


def retriever_agent(
    state: PipelineState,
    collection=None,
) -> PipelineState:
    """Perform hybrid retrieval: tone-matched + content-matched chunks.

    Tone chunks: filtered by tone metadata, used as style references.
    Content chunks: filtered by cosine similarity, used as personal context.
    """
    if not collection or collection.count() == 0:
        state.tone_chunks = []
        state.content_chunks = []
        return state

    query_text = state.resolved_content or state.raw_input

    # 1. Tone retrieval — find chunks where twin sounds like the detected tone
    tone_chunks = []
    if state.tone:
        similar_tones = get_similar_tones(state.tone)
        try:
            tone_results = collection.query(
                query_texts=[query_text],
                n_results=10,
                where={
                    "$and": [
                        {"tone": {"$in": similar_tones}},
                        {"twin_msg_ratio": {"$gte": 0.3}},
                    ]
                },
                include=["documents", "distances", "metadatas"],
            )
            for i in range(len(tone_results["ids"][0])):
                tone_chunks.append({
                    "chunk_id": tone_results["ids"][0][i],
                    "document": tone_results["documents"][0][i],
                    "distance": tone_results["distances"][0][i],
                    "metadata": tone_results["metadatas"][0][i],
                })
        except Exception:
            # Metadata filter may fail if chunks lack tone field — fall back to no filter
            pass

    # If tone query returned nothing (no enriched chunks), fall back to unfiltered
    if not tone_chunks:
        try:
            fallback = collection.query(
                query_texts=[query_text],
                n_results=TONE_RESULTS_COUNT,
                include=["documents", "distances", "metadatas"],
            )
            for i in range(len(fallback["ids"][0])):
                tone_chunks.append({
                    "chunk_id": fallback["ids"][0][i],
                    "document": fallback["documents"][0][i],
                    "distance": fallback["distances"][0][i],
                    "metadata": fallback["metadatas"][0][i],
                })
        except Exception:
            pass

    state.tone_chunks = tone_chunks[:TONE_RESULTS_COUNT]

    # 2. Content retrieval — find chunks about similar topics
    content_chunks = []
    try:
        content_results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={"twin_msg_ratio": {"$gte": 0.2}},
            include=["documents", "distances", "metadatas"],
        )
        for i in range(len(content_results["ids"][0])):
            dist = content_results["distances"][0][i]
            if dist < CONTENT_MAX_DISTANCE:
                content_chunks.append({
                    "chunk_id": content_results["ids"][0][i],
                    "document": content_results["documents"][0][i],
                    "distance": dist,
                    "metadata": content_results["metadatas"][0][i],
                })
    except Exception:
        pass

    # Deduplicate (a chunk could appear in both tone and content results)
    tone_ids = {c["chunk_id"] for c in state.tone_chunks}
    content_chunks = [c for c in content_chunks if c["chunk_id"] not in tone_ids]

    state.content_chunks = content_chunks[:CONTENT_RESULTS_COUNT]
    return state

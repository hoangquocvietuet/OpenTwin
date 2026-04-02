"""ChromaDB ingestion from train_chunks.jsonl.

Uses sentence-transformers for embeddings. The embedding model is configured
via EMBEDDING_MODEL in .env. Changing it requires re-embedding all chunks.
"""

import json

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def get_embedding_function(model_name: str) -> SentenceTransformerEmbeddingFunction:
    """Create a sentence-transformer embedding function for ChromaDB."""
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def load_chunks_from_jsonl(path: str) -> list[dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _chunk_to_document(chunk: dict) -> str:
    """Build the document text that gets embedded.

    Includes context + response so retrieval matches both
    the question pattern and the response style.
    """
    parts = []
    for ctx in chunk.get("context", []):
        parts.append(f"{ctx['author']}: {ctx['text']}")
    parts.append(f"{chunk['response']['author']}: {chunk['response']['text']}")
    return "\n".join(parts)


def ingest_chunks(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list[dict],
    embedding_function: SentenceTransformerEmbeddingFunction | None = None,
) -> chromadb.Collection:
    """Ingest chunks into a ChromaDB collection.

    Deletes existing collection if present (overwrite mode).
    embedding_function must be provided to ensure consistent embeddings.
    """
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )

    # Batch add
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(_chunk_to_document(chunk))
        metadatas.append({
            "chunk_type": chunk.get("chunk_type", ""),
            "score": chunk.get("score", 0.0),
            "source_thread": chunk.get("thread_id", ""),
            "timestamp": chunk.get("response", {}).get("timestamp", ""),
            "context_turns": chunk.get("context_turns", 0),
            "response_length": chunk.get("response_length", 0),
        })

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection

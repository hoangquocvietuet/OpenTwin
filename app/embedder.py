"""ChromaDB ingestion from train_chunks.jsonl.

Uses the OpenAI-compatible embeddings API (/v1/embeddings) so any provider
works: OpenAI, Ollama, OpenRouter, LiteLLM, etc. — just change base_url and api_key.
"""

import json
import logging
from typing import Union, cast

import openai
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function using the OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: float = 300,
    ):
        self._model_name = model_name
        self._base_url = base_url
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info("Embedding function: model=%s base_url=%s", model_name, base_url)

    def __call__(self, input: Union[Documents, str]) -> Embeddings:
        texts = input if isinstance(input, list) else [input]
        logger.info("Embedding %d texts via %s at %s", len(texts), self._model_name, self._base_url)
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=texts,
            )
            logger.info("Embedding done: %d vectors returned", len(response.data))
            return cast(Embeddings, [item.embedding for item in response.data])
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            raise


def get_embedding_function(
    model_name: str,
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> OpenAICompatibleEmbeddingFunction:
    """Create an embedding function for ChromaDB.

    Works with any OpenAI-compatible API: OpenAI, Ollama, OpenRouter, etc.
    """
    return OpenAICompatibleEmbeddingFunction(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )


def load_chunks_from_jsonl(path: str) -> list[dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


BOT_AUTHORS = {"Meta AI", "Facebook", "Instagram", "Messenger"}

LLM_MARKERS = [
    "Kính thưa", "Dưới đây là", "Tôi là Meta AI",
    "BÀI THU HOẠCH", "I. PHẦN MỞ ĐẦU", "II. CẢM NHẬN",
    "bài viết được thiết kế", "Gợi ý để bài viết",
]

MAX_RESPONSE_LENGTH = 500


def _is_bad_chunk(chunk: dict) -> bool:
    """Check if a chunk should be excluded from embedding."""
    for ctx in chunk.get("context", []):
        if ctx.get("author", "") in BOT_AUTHORS:
            return True
    if chunk.get("response", {}).get("author", "") in BOT_AUTHORS:
        return True

    resp_text = chunk.get("response", {}).get("text", "")

    if len(resp_text) > MAX_RESPONSE_LENGTH:
        return True

    if any(marker in resp_text for marker in LLM_MARKERS):
        return True

    for ctx in chunk.get("context", []):
        if len(ctx.get("text", "")) > MAX_RESPONSE_LENGTH * 2:
            if any(marker in ctx.get("text", "") for marker in LLM_MARKERS):
                return True

    return False


def _chunk_to_document(chunk: dict) -> str:
    """Build the document text that gets embedded."""
    parts = []
    for ctx in chunk.get("context", []):
        parts.append(f"{ctx['author']}: {ctx['text']}")
    parts.append(f"{chunk['response']['author']}: {chunk['response']['text']}")
    return "\n".join(parts)


BATCH_SIZE = 512


def ingest_chunks(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list[dict],
    embedding_function=None,
) -> chromadb.Collection:
    """Ingest chunks into a ChromaDB collection.

    Deletes existing collection if present (overwrite mode).
    Batches additions to avoid timeouts.
    """
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )

    ids = []
    documents = []
    metadatas = []

    skipped = 0
    for chunk in chunks:
        if _is_bad_chunk(chunk):
            skipped += 1
            continue
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

    # Batch add to avoid timeouts on large datasets
    for i in range(0, len(ids), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

    return collection

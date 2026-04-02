# app/rechunk.py
"""Rechunk CLI — re-run boundary detection and analyzers on raw data.

Usage:
    python -m app.rechunk
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import chromadb
import openai

from app.analyzers.default_registry import create_default_registry
from app.analyzers.registry import run_analyzers
from app.chunking.boundary import detect_boundaries
from app.chunking.normalizer import normalize_segments, build_chunks
from app.config import settings
from app.embedder import get_embedding_function
from app.importer import _safe_collection_name

logger = logging.getLogger(__name__)


def load_raw_messages_from_sources(
    data_dir: str,
    twin_slug: str,
) -> dict[str, list[dict]]:
    """Load raw messages from all enabled sources, grouped by thread.

    Reads train_chunks.jsonl from each enabled source and reconstructs
    per-thread message lists from the chunk context + response fields.

    Returns:
        Dict mapping thread_id → list of messages (sorted by timestamp).
    """
    twin_dir = os.path.join(data_dir, twin_slug)
    sources_file = os.path.join(twin_dir, "sources.json")

    if not os.path.isfile(sources_file):
        return {}

    with open(sources_file) as f:
        sources = json.load(f)

    messages_by_thread: dict[str, list[dict]] = defaultdict(list)
    seen_timestamps: dict[str, set[str]] = defaultdict(set)

    for source in sources:
        if not source.get("enabled", True):
            continue

        chunks_path = os.path.join(twin_dir, "sources", source["id"], "train_chunks.jsonl")
        if not os.path.isfile(chunks_path):
            continue

        with open(chunks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                thread_id = chunk.get("thread_id", "unknown")

                # Extract messages from context + response
                for ctx in chunk.get("context", []):
                    ts = ctx.get("timestamp", "")
                    if ts not in seen_timestamps[thread_id]:
                        messages_by_thread[thread_id].append({
                            "author": ctx.get("author", "?"),
                            "text": ctx.get("text", ""),
                            "timestamp": ts,
                        })
                        seen_timestamps[thread_id].add(ts)

                resp = chunk.get("response", {})
                if resp:
                    ts = resp.get("timestamp", "")
                    if ts not in seen_timestamps[thread_id]:
                        messages_by_thread[thread_id].append({
                            "author": resp.get("author", "?"),
                            "text": resp.get("text", ""),
                            "timestamp": ts,
                        })
                        seen_timestamps[thread_id].add(ts)

    # Sort each thread by timestamp
    for thread_id in messages_by_thread:
        messages_by_thread[thread_id].sort(key=lambda m: m.get("timestamp", ""))

    return dict(messages_by_thread)


def rechunk_twin(
    data_dir: str,
    twin_slug: str,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> list[dict]:
    """Re-chunk all raw messages for a twin using dynamic boundary detection.

    Args:
        data_dir: Base data directory
        twin_slug: Twin identifier
        twin_name: Twin display name
        llm_client: OpenAI-compatible client for boundary detection
        llm_model: Model name

    Returns:
        List of chunk dicts ready for enrichment and ChromaDB ingestion.
    """
    messages_by_thread = load_raw_messages_from_sources(data_dir, twin_slug)

    if not messages_by_thread:
        logger.warning("No raw messages found.")
        return []

    all_chunks = []
    for thread_id, messages in messages_by_thread.items():
        if len(messages) < 2:
            continue

        # Detect boundaries
        boundaries = detect_boundaries(messages, llm_client=llm_client, llm_model=llm_model)

        # Normalize segments
        segments = normalize_segments(messages, boundaries, min_size=3, max_size=20)

        # Build chunk dicts
        chunks = build_chunks(segments, thread_id=thread_id, twin_name=twin_name)
        all_chunks.extend(chunks)

    logger.info(f"Rechunked into {len(all_chunks)} chunks from {len(messages_by_thread)} threads.")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Re-chunk and re-analyze all data for a twin")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Setup
    chromadb_client = chromadb.PersistentClient(path=settings.chromadb_path)
    ef = get_embedding_function(
        settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )

    # Detect twin
    twin_slug = None
    for name in os.listdir(settings.data_dir):
        twin_dir = os.path.join(settings.data_dir, name)
        if os.path.isdir(twin_dir) and os.path.isfile(os.path.join(twin_dir, "sources.json")):
            twin_slug = name
            break

    if not twin_slug:
        print("No twin found.")
        sys.exit(1)

    twin_name = twin_slug.replace("_", " ").title()
    collection_name = _safe_collection_name(twin_slug)

    # LLM clients
    analyzer_client = openai.OpenAI(
        base_url=settings.analyzer_base_url,
        api_key=settings.analyzer_api_key,
    )

    # 1. Rechunk
    print(f"Rechunking {twin_slug}...")
    chunks = rechunk_twin(
        data_dir=settings.data_dir,
        twin_slug=twin_slug,
        twin_name=twin_name,
        llm_client=analyzer_client,
        llm_model=settings.analyzer_model,
    )

    if not chunks:
        print("No chunks produced.")
        sys.exit(1)

    # 2. Enrich with analyzers
    print(f"Enriching {len(chunks)} chunks with analyzers...")
    registry = create_default_registry()

    for i, chunk in enumerate(chunks):
        prev_chunk = chunks[i - 1] if i > 0 else None
        next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

        new_meta = run_analyzers(
            registry, chunk, twin_name=twin_name,
            prev_chunk=prev_chunk, next_chunk=next_chunk,
            llm_client=analyzer_client, llm_model=settings.analyzer_model,
        )
        chunk["metadata"].update(new_meta)

        # Serialize _analyzers_applied for ChromaDB
        if "_analyzers_applied" in chunk["metadata"]:
            chunk["metadata"]["_analyzers_applied"] = json.dumps(chunk["metadata"]["_analyzers_applied"])

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(chunks)} enriched")

    # 3. Delete old collection and create new with cosine
    print(f"Recreating collection '{collection_name}'...")
    try:
        chromadb_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chromadb_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )

    # 4. Ingest
    print(f"Ingesting {len(chunks)} chunks...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"Done. {collection.count()} chunks in collection.")


if __name__ == "__main__":
    main()

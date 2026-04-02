"""Import pipeline supporting multiple data formats.

Supported formats:
- Facebook zip export (inbox + e2ee_cutover + archived)
- E2EE messages folder (JSON files from Messenger E2EE export)

Each import creates a "source" under data/<twin>/sources/<source_id>/.
"""

import json
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from datetime import datetime, timezone
from typing import Callable

import chromadb

import audit_facebook
import score_and_chunk
from app.adapters import convert_e2ee_to_canonical, detect_format
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function
from app.sources import (
    Source, generate_source_id, register_source,
    load_sources, get_enabled_chunk_paths, _source_dir,
)


class ZipValidationError(Exception):
    pass


def _safe_collection_name(name: str) -> str:
    """Convert a name to a ChromaDB-safe collection name."""
    import hashlib
    suffix = hashlib.sha256(name.encode()).hexdigest()[:8]
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace(" ", "_")
    ascii_name = re.sub(r"[^a-zA-Z0-9_\-]", "", ascii_name)
    ascii_name = ascii_name.strip("_-")
    prefix = ascii_name[:50] if ascii_name else "twin"
    return f"{prefix}_{suffix}"


def validate_zip(path: str, max_size_mb: int = 500) -> bool:
    if not zipfile.is_zipfile(path):
        raise ZipValidationError("Please upload a .zip file")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ZipValidationError(f"File too large (max {max_size_mb}MB)")
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in name:
                raise ZipValidationError("Invalid zip file")
    return True


def find_inbox_folder(extracted_path: str) -> str | None:
    """Find and merge all message folders from a Facebook export."""
    messages_dir = None
    for dirpath, dirnames, _ in os.walk(extracted_path):
        if os.path.basename(dirpath) == "inbox":
            messages_dir = os.path.dirname(dirpath)
            break

    if not messages_dir:
        return None

    conv_folders = ["inbox", "e2ee_cutover", "archived_threads", "filtered_threads", "message_requests"]

    has_extra = any(
        os.path.isdir(os.path.join(messages_dir, f))
        for f in conv_folders if f != "inbox"
    )

    inbox_path = os.path.join(messages_dir, "inbox")
    if not has_extra:
        return inbox_path

    merged = tempfile.mkdtemp(prefix="dt_merged_inbox_")
    total = 0
    for folder in conv_folders:
        folder_path = os.path.join(messages_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for conv in os.listdir(folder_path):
            src = os.path.join(folder_path, conv)
            if not os.path.isdir(src):
                continue
            dest_name = conv if folder == "inbox" else f"{folder}__{conv}"
            dest = os.path.join(merged, dest_name)
            if not os.path.exists(dest):
                os.symlink(src, dest)
                total += 1

    if total == 0:
        shutil.rmtree(merged, ignore_errors=True)
        return inbox_path

    print(f"Merged {total} conversation folders")
    return merged


def _get_or_create_collection(chromadb_client, twin_slug, embedding_model,
                              embedding_base_url="http://localhost:11434/v1",
                              embedding_api_key="ollama"):
    """Get or create the ChromaDB collection for a twin."""
    collection_name = _safe_collection_name(twin_slug)
    ef = get_embedding_function(embedding_model, base_url=embedding_base_url, api_key=embedding_api_key)
    return chromadb_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )


def add_source_embeddings(
    data_dir: str,
    twin_slug: str,
    source_id: str,
    chromadb_client: chromadb.ClientAPI,
    embedding_model: str = "text-embedding-3-small",
    on_progress: Callable[[str], None] | None = None,
    embedding_base_url: str = "http://localhost:11434/v1",
    embedding_api_key: str = "ollama",
) -> int:
    """Add a single source's chunks to ChromaDB. Returns count added."""
    from app.sources import _source_dir

    collection = _get_or_create_collection(
        chromadb_client, twin_slug, embedding_model, embedding_base_url, embedding_api_key
    )

    chunks_path = os.path.join(_source_dir(data_dir, twin_slug, source_id), "train_chunks.jsonl")
    if not os.path.isfile(chunks_path):
        return 0

    chunks = load_chunks_from_jsonl(chunks_path)

    # Prefix IDs with source_id
    for chunk in chunks:
        chunk["chunk_id"] = f"{source_id}/{chunk['chunk_id']}"

    if on_progress:
        on_progress(f"Embedding {len(chunks)} chunks from source {source_id}...")

    # Use ingest logic but add to existing collection instead of replacing
    from app.embedder import _is_bad_chunk, _chunk_to_document, BATCH_SIZE

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        if _is_bad_chunk(chunk):
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

    total = len(ids)
    for i in range(0, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        if on_progress:
            on_progress(f"Embedding batch {i + 1}–{batch_end} of {total} chunks...")
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

    return total


def remove_source_embeddings(
    twin_slug: str,
    source_id: str,
    chromadb_client: chromadb.ClientAPI,
    embedding_model: str = "text-embedding-3-small",
    on_progress: Callable[[str], None] | None = None,
    embedding_base_url: str = "http://localhost:11434/v1",
    embedding_api_key: str = "ollama",
) -> int:
    """Remove a single source's chunks from ChromaDB. Returns count removed."""
    collection_name = _safe_collection_name(twin_slug)
    ef = get_embedding_function(embedding_model, base_url=embedding_base_url, api_key=embedding_api_key)

    try:
        collection = chromadb_client.get_collection(collection_name, embedding_function=ef)
    except ValueError:
        return 0

    if on_progress:
        on_progress(f"Removing chunks from source {source_id}...")

    # Get all IDs that start with this source_id prefix
    prefix = f"{source_id}/"
    all_ids = collection.get(include=[])["ids"]
    to_remove = [id for id in all_ids if id.startswith(prefix)]

    if to_remove:
        # ChromaDB delete has a batch limit
        for i in range(0, len(to_remove), 5000):
            collection.delete(ids=to_remove[i:i + 5000])

    return len(to_remove)


def rebuild_embeddings(
    data_dir: str,
    twin_slug: str,
    chromadb_client: chromadb.ClientAPI,
    embedding_model: str = "text-embedding-3-small",
    on_progress: Callable[[str], None] | None = None,
    embedding_base_url: str = "http://localhost:11434/v1",
    embedding_api_key: str = "ollama",
) -> int:
    """Full rebuild — only needed when switching embedding models."""
    collection_name = _safe_collection_name(twin_slug)
    ef = get_embedding_function(embedding_model, base_url=embedding_base_url, api_key=embedding_api_key)

    # Delete and recreate
    try:
        chromadb_client.delete_collection(collection_name)
    except ValueError:
        pass

    source_entries = get_enabled_chunk_paths(data_dir, twin_slug)
    total = 0
    for source_id, path in source_entries:
        count = add_source_embeddings(
            data_dir, twin_slug, source_id, chromadb_client, embedding_model, on_progress,
            embedding_base_url=embedding_base_url, embedding_api_key=embedding_api_key,
        )
        total += count

    return total


def _chunk_and_save(
    canonical: list[dict],
    source_dir: str,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[int, int, str]:
    """Chunk canonical messages, build fingerprint, save to source_dir.

    Returns (train_count, holdout_count, fingerprint_path).
    """
    # Save canonical messages
    jsonl_path = os.path.join(source_dir, "cleaned_messages.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for msg in canonical:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    if on_progress:
        on_progress("Chunking and scoring...")

    messages = score_and_chunk.load_jsonl(jsonl_path)
    threads = score_and_chunk.group_by_thread(messages)

    all_chunks = []
    for tid, thread_msgs in threads.items():
        if score_and_chunk.is_dm_thread(thread_msgs):
            chunks = score_and_chunk.extract_dm_chunks(thread_msgs, tid)
        else:
            chunks = score_and_chunk.extract_group_chunks(thread_msgs, tid)
        all_chunks.extend(chunks)

    for chunk in all_chunks:
        score_and_chunk.score_chunk(chunk)

    fp = score_and_chunk.build_fingerprint(messages)

    train, holdout = score_and_chunk.stratified_holdout(
        sorted(all_chunks, key=lambda c: c.score, reverse=True)
    )

    train_path = os.path.join(source_dir, "train_chunks.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for chunk in train:
            f.write(json.dumps(score_and_chunk.chunk_to_dict(chunk), ensure_ascii=False) + "\n")

    holdout_path = os.path.join(source_dir, "holdout_chunks.jsonl")
    with open(holdout_path, "w", encoding="utf-8") as f:
        for chunk in holdout:
            f.write(json.dumps(score_and_chunk.chunk_to_dict(chunk), ensure_ascii=False) + "\n")

    fp_dict = {
        "total_messages": fp.total_messages,
        "avg_length": fp.avg_length,
        "median_length": fp.median_length,
        "avg_words_per_msg": fp.avg_words_per_msg,
        "length_distribution": fp.length_distribution,
        "punctuation": {
            "all_lowercase_pct": fp.all_lowercase_pct,
            "ends_with_period_pct": fp.ends_with_period_pct,
            "uses_exclamation_pct": fp.uses_exclamation_pct,
            "question_mark_pct": fp.question_mark_pct,
            "uses_ellipsis_pct": fp.uses_ellipsis_pct,
            "has_emoji_pct": fp.ends_with_emoji_pct,
        },
        "top_emojis": fp.top_emojis,
        "top_words": fp.top_words,
    }
    fp_path = os.path.join(source_dir, "style_fingerprint.json")
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(fp_dict, f, indent=2, ensure_ascii=False)

    return len(train), len(holdout), fp_path


def _enrich_chunks_with_analyzers(chunks, twin_name, analyzer_base_url=None,
                                   analyzer_model=None, analyzer_api_key=None):
    """Optionally enrich chunks with analyzer metadata.

    Returns enriched chunks if analyzer model is configured, else returns chunks as-is.
    """
    if not analyzer_base_url or not analyzer_model:
        return chunks

    try:
        import openai as openai_module
        from app.analyzers.default_registry import create_default_registry
        from app.analyzers.registry import run_analyzers

        client = openai_module.OpenAI(base_url=analyzer_base_url, api_key=analyzer_api_key or "ollama")
        registry = create_default_registry()

        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

            new_meta = run_analyzers(
                registry, chunk, twin_name=twin_name,
                prev_chunk=prev_chunk, next_chunk=next_chunk,
                llm_client=client, llm_model=analyzer_model,
            )
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            chunk["metadata"].update(new_meta)

        return chunks
    except ImportError:
        return chunks


def run_import_pipeline(
    zip_path: str,
    chromadb_client: chromadb.ClientAPI,
    data_dir: str,
    embedding_model: str = "text-embedding-3-small",
    target_name: str | None = None,
    source_name: str = "",
    on_progress: Callable[[str], None] | None = None,
    embedding_base_url: str = "http://localhost:11434/v1",
    embedding_api_key: str = "ollama",
) -> dict:
    """Import a Facebook zip export. Creates a new source."""
    if on_progress:
        on_progress("Validating zip file...")

    validate_zip(zip_path)

    tmp_dir = tempfile.mkdtemp(prefix="dt_import_")
    try:
        if on_progress:
            on_progress("Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Check if the zip contains E2EE messages (JSON files with participants)
        fmt = detect_format(tmp_dir)
        if fmt == "e2ee_messages":
            return run_e2ee_import(
                folder_path=tmp_dir,
                chromadb_client=chromadb_client,
                data_dir=data_dir,
                embedding_model=embedding_model,
                target_name=target_name,
                source_name=source_name or "E2EE messages (zip)",
                on_progress=on_progress,
                _cleanup_folder=False,  # tmp_dir cleaned in finally
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
            )

        inbox_path = find_inbox_folder(tmp_dir)
        if not inbox_path:
            raise ValueError("No Facebook messages found in this export")

        if on_progress:
            on_progress("Parsing messages...")
        try:
            report, canonical = audit_facebook.run_audit(
                inbox_path, target_name=target_name,
            )
        except SystemExit:
            raise ValueError("Failed to parse Facebook data. Check the export format.")

        if report.target_messages == 0:
            raise ValueError("No conversations found")

        safe_name = report.target_name.replace(" ", "_").lower()
        source_id = generate_source_id()
        source_dir = _source_dir(data_dir, safe_name, source_id)
        os.makedirs(source_dir, exist_ok=True)

        # Save audit report
        report_data = {
            "target_name": report.target_name,
            "is_self": report.is_self,
            "total_conversations": report.total_conversations,
            "dm_chats": report.dm_chats,
            "group_chats": report.group_chats,
            "total_messages": report.total_messages,
            "target_messages": report.target_messages,
            "type_counts": report.type_counts,
            "length_buckets": report.length_buckets,
            "messages_by_month": report.messages_by_month,
            "top_conversations": report.conversations[:50],
        }
        with open(os.path.join(source_dir, "audit_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Save canonical messages
        canon_list = []
        for msg in canonical:
            canon_list.append({
                "id": msg["id"],
                "source": msg["source"],
                "timestamp": msg["timestamp"],
                "thread_id": msg["thread_id"],
                "author": msg["author"],
                "is_target": msg["is_target"],
                "text": msg["text"],
                "msg_type": msg["msg_type"],
                "reactions": msg["reactions"],
                "metadata": msg["metadata"],
            })

        train_count, holdout_count, fp_path = _chunk_and_save(
            canon_list, source_dir, on_progress
        )

        auto_name = source_name or f"Facebook export {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        source = Source(
            id=source_id,
            name=auto_name,
            platform="facebook",
            twin_slug=safe_name,
            twin_name=report.target_name,
            enabled=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_messages=report.total_messages,
            target_messages=report.target_messages,
            train_chunks=train_count,
            holdout_chunks=holdout_count,
            dm_chats=report.dm_chats,
            group_chats=report.group_chats,
        )
        register_source(data_dir, source)

        if on_progress:
            on_progress("Embedding new source...")
        source_embedded = add_source_embeddings(
            data_dir, safe_name, source_id, chromadb_client, embedding_model, on_progress,
            embedding_base_url=embedding_base_url, embedding_api_key=embedding_api_key,
        )

        return {
            "status": "success",
            "source_id": source_id,
            "twin_slug": safe_name,
            "twin_name": report.target_name,
            "total_messages": report.target_messages,
            "chunks_embedded": source_embedded,
            "source_chunks": train_count,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_e2ee_import(
    folder_path: str,
    chromadb_client: chromadb.ClientAPI,
    data_dir: str,
    embedding_model: str = "text-embedding-3-small",
    target_name: str | None = None,
    source_name: str = "",
    on_progress: Callable[[str], None] | None = None,
    _cleanup_folder: bool = False,
    embedding_base_url: str = "http://localhost:11434/v1",
    embedding_api_key: str = "ollama",
) -> dict:
    """Import E2EE messages folder. Creates a new source."""
    try:
        if on_progress:
            on_progress("Converting E2EE messages...")

        canonical, detected_name = convert_e2ee_to_canonical(folder_path, target_name)

        if not canonical:
            raise ValueError("No messages found in the E2EE export")

        safe_name = detected_name.replace(" ", "_").lower()
        source_id = generate_source_id()
        source_dir = _source_dir(data_dir, safe_name, source_id)
        os.makedirs(source_dir, exist_ok=True)

        target_msgs = sum(1 for m in canonical if m["is_target"])
        total_msgs = len(canonical)
        dm_count = len(set(
            m["thread_id"] for m in canonical
            if m.get("metadata", {}).get("is_dm", False)
        ))
        group_count = len(set(
            m["thread_id"] for m in canonical
            if not m.get("metadata", {}).get("is_dm", True)
        ))

        # Save audit summary
        audit_data = {
            "target_name": detected_name,
            "is_self": True,
            "total_messages": total_msgs,
            "target_messages": target_msgs,
            "dm_chats": dm_count,
            "group_chats": group_count,
            "platform": "e2ee_messages",
        }
        with open(os.path.join(source_dir, "audit_report.json"), "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

        train_count, holdout_count, fp_path = _chunk_and_save(
            canonical, source_dir, on_progress
        )

        auto_name = source_name or f"E2EE messages {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        source = Source(
            id=source_id,
            name=auto_name,
            platform="e2ee_messages",
            twin_slug=safe_name,
            twin_name=detected_name,
            enabled=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_messages=total_msgs,
            target_messages=target_msgs,
            train_chunks=train_count,
            holdout_chunks=holdout_count,
            dm_chats=dm_count,
            group_chats=group_count,
        )
        register_source(data_dir, source)

        if on_progress:
            on_progress("Embedding new source...")
        source_embedded = add_source_embeddings(
            data_dir, safe_name, source_id, chromadb_client, embedding_model, on_progress,
            embedding_base_url=embedding_base_url, embedding_api_key=embedding_api_key,
        )

        return {
            "status": "success",
            "source_id": source_id,
            "twin_slug": safe_name,
            "twin_name": detected_name,
            "total_messages": target_msgs,
            "chunks_embedded": source_embedded,
            "source_chunks": train_count,
        }
    finally:
        if _cleanup_folder:
            shutil.rmtree(folder_path, ignore_errors=True)

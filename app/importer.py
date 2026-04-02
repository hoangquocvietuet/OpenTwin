"""Import pipeline: zip upload -> audit -> chunk -> embed into ChromaDB.

data_dir is the directory containing twin subdirectories directly,
e.g., ./data/hoang_quoc_viet/. This is the same DATA_DIR from .env.
"""

import json
import os
import shutil
import tempfile
import zipfile
from typing import Callable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import existing scripts as modules
import audit_facebook
import score_and_chunk
from app.embedder import ingest_chunks, get_embedding_function


class ZipValidationError(Exception):
    pass


def validate_zip(path: str, max_size_mb: int = 500) -> bool:
    """Validate that a file is a valid zip under the size limit."""
    if not zipfile.is_zipfile(path):
        raise ZipValidationError("Please upload a .zip file")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ZipValidationError(f"File too large (max {max_size_mb}MB)")

    # Check for zip slip
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in name:
                raise ZipValidationError("Invalid zip file")

    return True


def find_inbox_folder(extracted_path: str) -> str | None:
    """Find the inbox/ folder in extracted zip contents.

    Checks up to 3 levels deep to handle various export structures:
    - inbox/
    - messages/inbox/
    - your_facebook_activity/messages/inbox/
    """
    # Check direct inbox/
    inbox = os.path.join(extracted_path, "inbox")
    if os.path.isdir(inbox):
        return inbox

    # Check one level deep
    for item in os.listdir(extracted_path):
        candidate = os.path.join(extracted_path, item, "inbox")
        if os.path.isdir(candidate):
            return candidate
        # Check two levels deep
        sub = os.path.join(extracted_path, item)
        if os.path.isdir(sub):
            for sub_item in os.listdir(sub):
                candidate = os.path.join(sub, sub_item, "inbox")
                if os.path.isdir(candidate):
                    return candidate

    return None


def run_import_pipeline(
    zip_path: str,
    chromadb_client: chromadb.ClientAPI,
    data_dir: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    target_name: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Run the full import pipeline.

    Args:
        zip_path: Path to the uploaded .zip file.
        chromadb_client: ChromaDB persistent client.
        data_dir: The DATA_DIR — twin subdirectories live directly inside this.
                  e.g., data_dir="./data" → outputs go to ./data/<safe_name>/
        embedding_model: Name of the sentence-transformer model for ChromaDB.
        target_name: Name of the person to build a twin of (auto-detect if None).
        on_progress: Optional callback for progress updates.

    Returns dict with status and stats.
    """
    if on_progress:
        on_progress("Validating zip file...")

    validate_zip(zip_path)

    # Unzip to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="dt_import_")
    try:
        if on_progress:
            on_progress("Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Find inbox
        inbox_path = find_inbox_folder(tmp_dir)
        if not inbox_path:
            raise ValueError("No Facebook messages found in this export")

        # Run audit
        if on_progress:
            on_progress("Parsing messages...")
        try:
            report, canonical = audit_facebook.run_audit(
                inbox_path,
                target_name=target_name,
            )
        except SystemExit:
            raise ValueError("Failed to parse Facebook data. Check the export format.")

        if report.target_messages == 0:
            raise ValueError("No conversations found")

        safe_name = report.target_name.replace(" ", "_").lower()
        target_dir = os.path.join(data_dir, safe_name)
        os.makedirs(target_dir, exist_ok=True)

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
        with open(os.path.join(target_dir, "audit_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Save cleaned JSONL
        jsonl_path = os.path.join(target_dir, "cleaned_messages.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for msg in canonical:
                msg_out = {
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
                }
                f.write(json.dumps(msg_out, ensure_ascii=False) + "\n")

        # Run chunker
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

        # Build fingerprint
        fp = score_and_chunk.build_fingerprint(messages)

        # Save chunks and fingerprint
        train, holdout = score_and_chunk.stratified_holdout(
            sorted(all_chunks, key=lambda c: c.score, reverse=True)
        )

        train_path = os.path.join(target_dir, "train_chunks.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for chunk in train:
                f.write(json.dumps(score_and_chunk.chunk_to_dict(chunk), ensure_ascii=False) + "\n")

        holdout_path = os.path.join(target_dir, "holdout_chunks.jsonl")
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
        fp_path = os.path.join(target_dir, "style_fingerprint.json")
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fp_dict, f, indent=2, ensure_ascii=False)

        # Check if collection already exists
        collection_exists = False
        try:
            existing = chromadb_client.get_collection(safe_name)
            collection_exists = existing.count() > 0
        except ValueError:
            pass

        # Embed into ChromaDB
        if on_progress:
            on_progress(f"Embedding {len(train)} chunks...")
        from app.embedder import load_chunks_from_jsonl
        ef = get_embedding_function(embedding_model)
        chunk_dicts = load_chunks_from_jsonl(train_path)
        ingest_chunks(chromadb_client, safe_name, chunk_dicts, embedding_function=ef)

        if on_progress:
            on_progress("Twin ready!")

        return {
            "status": "success",
            "twin_slug": safe_name,
            "twin_name": report.target_name,
            "total_messages": report.target_messages,
            "chunks_embedded": len(train),
            "fingerprint_path": fp_path,
            "overwritten": collection_exists,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

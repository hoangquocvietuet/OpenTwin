"""Source management for imported data.

Each import creates a "source" — a self-contained dataset with its own
chunks, fingerprint, and metadata. Sources can be enabled/disabled/deleted.
ChromaDB is rebuilt from all enabled sources.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict


@dataclass
class Source:
    id: str
    name: str  # human-readable label (e.g. "Facebook main export")
    platform: str  # "facebook", "facebook_e2ee", etc.
    twin_slug: str
    twin_name: str
    enabled: bool = True
    created_at: str = ""
    total_messages: int = 0
    target_messages: int = 0
    train_chunks: int = 0
    holdout_chunks: int = 0
    dm_chats: int = 0
    group_chats: int = 0


def _sources_path(data_dir: str, twin_slug: str) -> str:
    return os.path.join(data_dir, twin_slug, "sources.json")


def _source_dir(data_dir: str, twin_slug: str, source_id: str) -> str:
    return os.path.join(data_dir, twin_slug, "sources", source_id)


def load_sources(data_dir: str, twin_slug: str) -> list[Source]:
    """Load all sources for a twin."""
    path = _sources_path(data_dir, twin_slug)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Source(**s) for s in data]


def save_sources(data_dir: str, twin_slug: str, sources: list[Source]):
    """Save sources manifest."""
    path = _sources_path(data_dir, twin_slug)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in sources], f, indent=2, ensure_ascii=False)


def generate_source_id() -> str:
    """Generate a unique source ID based on timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def register_source(data_dir: str, source: Source) -> Source:
    """Register a new source in the manifest."""
    sources = load_sources(data_dir, source.twin_slug)
    sources.append(source)
    save_sources(data_dir, source.twin_slug, sources)
    return source


def toggle_source(data_dir: str, twin_slug: str, source_id: str, enabled: bool) -> bool:
    """Enable or disable a source. Returns True if found."""
    sources = load_sources(data_dir, twin_slug)
    for s in sources:
        if s.id == source_id:
            s.enabled = enabled
            save_sources(data_dir, twin_slug, sources)
            return True
    return False


def delete_source(data_dir: str, twin_slug: str, source_id: str) -> bool:
    """Delete a source and its data. Returns True if found."""
    sources = load_sources(data_dir, twin_slug)
    new_sources = [s for s in sources if s.id != source_id]
    if len(new_sources) == len(sources):
        return False

    # Remove source data directory
    source_dir = _source_dir(data_dir, twin_slug, source_id)
    if os.path.isdir(source_dir):
        shutil.rmtree(source_dir)

    save_sources(data_dir, twin_slug, new_sources)
    return True


def get_enabled_chunk_paths(data_dir: str, twin_slug: str) -> list[tuple[str, str]]:
    """Get (source_id, path) tuples for train_chunks.jsonl of all enabled sources."""
    sources = load_sources(data_dir, twin_slug)
    result = []
    for s in sources:
        if not s.enabled:
            continue
        chunks_path = os.path.join(
            _source_dir(data_dir, s.twin_slug, s.id), "train_chunks.jsonl"
        )
        if os.path.isfile(chunks_path):
            result.append((s.id, chunks_path))
    return result


def get_merged_fingerprint_path(data_dir: str, twin_slug: str) -> str | None:
    """Get the fingerprint from the most recent enabled source."""
    sources = load_sources(data_dir, twin_slug)
    enabled = [s for s in sources if s.enabled]
    if not enabled:
        return None
    # Use the source with the most target messages
    best = max(enabled, key=lambda s: s.target_messages)
    fp_path = os.path.join(
        _source_dir(data_dir, twin_slug, best.id), "style_fingerprint.json"
    )
    return fp_path if os.path.isfile(fp_path) else None


def migrate_legacy_data(data_dir: str, twin_slug: str) -> bool:
    """Migrate old flat data structure to source-based structure.

    Old: data/<twin>/train_chunks.jsonl, style_fingerprint.json, etc.
    New: data/<twin>/sources/<source_id>/train_chunks.jsonl, etc.

    Returns True if migration happened.
    """
    twin_dir = os.path.join(data_dir, twin_slug)
    old_chunks = os.path.join(twin_dir, "train_chunks.jsonl")
    sources_file = _sources_path(data_dir, twin_slug)

    # Skip if already migrated or no old data
    if os.path.isfile(sources_file) or not os.path.isfile(old_chunks):
        return False

    source_id = "legacy_import"
    source_dir = _source_dir(data_dir, twin_slug, source_id)
    os.makedirs(source_dir, exist_ok=True)

    # Move data files to source directory
    files_to_move = [
        "train_chunks.jsonl", "holdout_chunks.jsonl",
        "style_fingerprint.json", "audit_report.json",
        "cleaned_messages.jsonl",
    ]
    for fname in files_to_move:
        src = os.path.join(twin_dir, fname)
        dst = os.path.join(source_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, dst)

    # Count stats from the moved data
    train_count = 0
    holdout_count = 0
    chunks_path = os.path.join(source_dir, "train_chunks.jsonl")
    if os.path.isfile(chunks_path):
        with open(chunks_path) as f:
            train_count = sum(1 for _ in f)
    holdout_path = os.path.join(source_dir, "holdout_chunks.jsonl")
    if os.path.isfile(holdout_path):
        with open(holdout_path) as f:
            holdout_count = sum(1 for _ in f)

    # Read audit report for stats
    total_msgs = 0
    target_msgs = 0
    dm_chats = 0
    group_chats = 0
    twin_name = twin_slug.replace("_", " ").title()
    audit_path = os.path.join(source_dir, "audit_report.json")
    if os.path.isfile(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)
        total_msgs = audit.get("total_messages", 0)
        target_msgs = audit.get("target_messages", 0)
        dm_chats = audit.get("dm_chats", 0)
        group_chats = audit.get("group_chats", 0)
        twin_name = audit.get("target_name", twin_name)

    source = Source(
        id=source_id,
        name="Legacy import (migrated)",
        platform="facebook",
        twin_slug=twin_slug,
        twin_name=twin_name,
        enabled=True,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_messages=total_msgs,
        target_messages=target_msgs,
        train_chunks=train_count,
        holdout_chunks=holdout_count,
        dm_chats=dm_chats,
        group_chats=group_chats,
    )
    register_source(data_dir, source)
    print(f"Migrated legacy data to source '{source_id}' ({train_count} chunks)")
    return True

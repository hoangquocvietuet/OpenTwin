"""Data format adapters for different chat export sources.

Each adapter converts a source format into the canonical message format
expected by score_and_chunk.py:
  {id, source, timestamp, thread_id, author, is_target, text, msg_type, reactions, metadata}
"""

import json
import os
from datetime import datetime, timezone


def detect_format(path: str) -> str:
    """Detect the format of an uploaded file/folder.

    Returns: "facebook_zip", "e2ee_messages", or "unknown"
    """
    if os.path.isfile(path) and path.endswith(".zip"):
        return "facebook_zip"

    if os.path.isdir(path):
        # Check if it looks like E2EE messages folder (JSON files with participants/messages)
        json_files = [f for f in os.listdir(path) if f.endswith(".json")]
        if json_files:
            sample = os.path.join(path, json_files[0])
            try:
                with open(sample) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "participants" in data and "messages" in data:
                    return "e2ee_messages"
            except (json.JSONDecodeError, KeyError):
                pass

    return "unknown"


def convert_e2ee_to_canonical(
    folder_path: str,
    target_name: str | None = None,
) -> tuple[list[dict], str]:
    """Convert E2EE message JSONs to canonical format.

    Args:
        folder_path: Path to folder containing <Name>_<id>.json files
        target_name: Name of the twin target (auto-detect if None)

    Returns:
        (canonical_messages, detected_target_name)
    """
    json_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".json") and os.path.isfile(os.path.join(folder_path, f))
    ])

    if not json_files:
        raise ValueError("No JSON files found in the messages folder")

    # Auto-detect target: most frequent sender across all conversations
    if not target_name:
        from collections import Counter
        sender_counts = Counter()
        for fname in json_files:
            with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                data = json.load(f)
            for msg in data.get("messages", []):
                if msg.get("senderName"):
                    sender_counts[msg["senderName"]] += 1
        if sender_counts:
            target_name = sender_counts.most_common(1)[0][0]
        else:
            raise ValueError("Could not detect target name from messages")

    print(f"E2EE adapter: target = {target_name}")

    canonical = []
    msg_id = 0

    for fname in json_files:
        filepath = os.path.join(folder_path, fname)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        thread_name = data.get("threadName", fname.replace(".json", ""))
        participants = data.get("participants", [])
        messages = data.get("messages", [])

        # Determine if DM or group
        is_dm = len(participants) == 2

        for msg in messages:
            # Skip unsent messages
            if msg.get("isUnsent", False):
                continue

            sender = msg.get("senderName", "")
            text = msg.get("text", "")
            msg_type = msg.get("type", "text")
            timestamp_ms = msg.get("timestamp", 0)

            # Skip non-text messages with no text
            if not text and msg_type != "text":
                continue
            if not text:
                continue

            # Convert timestamp
            try:
                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                ts_str = dt.isoformat()
            except (ValueError, OSError):
                ts_str = ""

            # Reactions
            reactions = []
            for r in msg.get("reactions", []):
                if isinstance(r, dict):
                    reactions.append({
                        "reaction": r.get("reaction", ""),
                        "actor": r.get("senderName", r.get("actor", "")),
                    })

            canonical.append({
                "id": f"e2ee_{msg_id}",
                "source": f"e2ee/{thread_name}",
                "timestamp": ts_str,
                "thread_id": f"e2ee/{thread_name}",
                "author": sender,
                "is_target": sender == target_name,
                "text": text,
                "msg_type": "text" if msg_type == "text" else msg_type,
                "reactions": reactions,
                "metadata": {
                    "is_dm": is_dm,
                    "participants": participants,
                },
            })
            msg_id += 1

    # Sort by timestamp
    canonical.sort(key=lambda m: m["timestamp"])

    target_msgs = sum(1 for m in canonical if m["is_target"])
    print(f"E2EE adapter: {len(canonical)} messages ({target_msgs} from target)")

    return canonical, target_name

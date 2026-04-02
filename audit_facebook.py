"""
Facebook Messenger Data Audit & Parser

Parses Facebook JSON export, fixes mojibake encoding, and produces:
1. A statistical audit report (terminal + JSON)
2. Cleaned structured JSONL for the downstream pipeline

Supports multi-target mode: audit yourself OR any person you've chatted with.

Usage:
    python audit_facebook.py /path/to/messages/inbox
    python audit_facebook.py /path/to/messages/inbox --target "Person Name"
    python audit_facebook.py /path/to/messages/inbox --list-people

Output:
    ./data/<target_name>/audit_report.json
    ./data/<target_name>/cleaned_messages.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# =============================================================================
# Encoding fix
# =============================================================================

def fix_mojibake(text: str) -> str:
    """Fix Facebook's double-encoded UTF-8.

    Facebook exports encode UTF-8 bytes as Latin-1 code points in JSON
    unicode escapes. Reverse: encode as Latin-1 → decode as UTF-8.
    """
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


# =============================================================================
# Filters
# =============================================================================

SYSTEM_PATTERNS = [
    "set the nickname",
    "changed the group photo",
    "named the group",
    "created the group",
    "left the group",
    "changed the theme",
    "set the emoji",
    "joined the video chat",
    "started a video chat",
    "started sharing video",
    "ended the call",
    "pinned a message",
    "đã bày tỏ cảm xúc",
    "reacted",
]


def is_system_message(content: str) -> bool:
    content_lower = content.lower()
    return any(p in content_lower for p in SYSTEM_PATTERNS)


def is_media_only(msg: dict) -> bool:
    has_text = bool(msg.get("content", "").strip())
    has_media = any(
        msg.get(k)
        for k in ["photos", "videos", "gifs", "sticker", "audio_files", "files"]
    )
    return has_media and not has_text


def msg_type(msg: dict) -> str:
    """Classify message type for audit stats."""
    content = fix_mojibake(msg.get("content", ""))
    if is_media_only(msg):
        return "media_only"
    if not content.strip():
        return "empty"
    if is_system_message(content):
        return "system"
    if len(content.strip()) <= 5:
        return "short"
    if msg.get("share"):
        return "link_share"
    return "text"


# =============================================================================
# Canonical message schema
# =============================================================================

def to_canonical(msg: dict, thread_id: str, source_file: str, target_name: str) -> dict:
    """Convert a raw Facebook message to the canonical pipeline format.

    target_name: the person whose twin we're building (could be self or another person).
    """
    sender = fix_mojibake(msg.get("sender_name", ""))
    content = fix_mojibake(msg.get("content", ""))
    ts_ms = msg.get("timestamp_ms", 0)

    reactions = []
    for r in msg.get("reactions", []):
        reactions.append({
            "emoji": fix_mojibake(r.get("reaction", "")),
            "actor": fix_mojibake(r.get("actor", "")),
        })

    return {
        "id": f"fb_{thread_id}_{ts_ms}",
        "source": "facebook",
        "timestamp": datetime.fromtimestamp(ts_ms / 1000).isoformat() if ts_ms else None,
        "timestamp_ms": ts_ms,
        "thread_id": thread_id,
        "author": sender,
        "is_target": sender == target_name,
        "text": content,
        "reply_to": None,  # FB export doesn't expose reply threading
        "reactions": reactions,
        "msg_type": msg_type(msg),
        "metadata": {
            "source_file": source_file,
            "has_photos": bool(msg.get("photos")),
            "has_videos": bool(msg.get("videos")),
            "has_sticker": bool(msg.get("sticker")),
            "has_share": bool(msg.get("share")),
        },
    }


# =============================================================================
# Name detection
# =============================================================================

def detect_self_name(inbox_path: Path) -> str:
    """Detect user's name = person who appears in the most conversations.

    The data owner is a participant in every conversation. Others only
    appear in some. Counting conversations (not messages) avoids bias
    from chatty friends in a few threads.
    """
    # count how many conversations each person participates in
    participation = Counter()
    for conv_dir in inbox_path.iterdir():
        if not conv_dir.is_dir():
            continue
        for msg_file in sorted(conv_dir.glob("message_*.json"))[:1]:
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for p in data.get("participants", []):
                name = fix_mojibake(p.get("name", ""))
                if name:
                    participation[name] += 1
            break  # only need first file per conversation for participants
    return participation.most_common(1)[0][0] if participation else ""


# =============================================================================
# Audit
# =============================================================================

def list_all_people(inbox_path: Path) -> list[tuple[str, int]]:
    """List all people found in the inbox with their message counts."""
    sender_counts = Counter()
    conv_dirs = sorted([d for d in inbox_path.iterdir() if d.is_dir()])
    for conv_dir in conv_dirs:
        for msg_file in sorted(conv_dir.glob("message_*.json")):
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for msg in data.get("messages", []):
                sender = fix_mojibake(msg.get("sender_name", ""))
                if sender:
                    sender_counts[sender] += 1
    return sender_counts.most_common()


@dataclass
class AuditReport:
    target_name: str = ""
    is_self: bool = True  # True if target is the data owner, False if another person
    inbox_path: str = ""
    total_conversations: int = 0
    dm_chats: int = 0
    group_chats: int = 0
    total_messages: int = 0
    target_messages: int = 0
    # quality breakdown (target's messages only)
    type_counts: dict = field(default_factory=dict)
    length_buckets: dict = field(default_factory=dict)
    messages_by_month: dict = field(default_factory=dict)
    # top conversations
    conversations: list = field(default_factory=list)


def length_bucket(n: int) -> str:
    if n == 0:
        return "empty"
    if n <= 5:
        return "1-5"
    if n <= 20:
        return "6-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    if n <= 300:
        return "101-300"
    return "300+"


def run_audit(inbox_path: str, self_name: str | None = None, target_name: str | None = None):
    inbox = Path(inbox_path)
    if not inbox.is_dir():
        print(f"Error: {inbox_path} is not a directory")
        sys.exit(1)

    report = AuditReport(inbox_path=inbox_path)

    # determine data owner
    if self_name:
        detected_self = self_name
        print(f"Data owner: {detected_self} (provided via --self)")
    else:
        detected_self = detect_self_name(inbox)
        print(f"Data owner: {detected_self} (auto-detected)")

    # determine target
    if target_name:
        report.target_name = target_name
        report.is_self = (target_name == detected_self)
    else:
        report.target_name = detected_self
        report.is_self = True

    label = "yourself" if report.is_self else "other person"
    print(f"Target: {report.target_name} ({label})")
    if not report.is_self:
        print(f"  (data owner: {detected_self})")

    type_counts = Counter()
    len_buckets = Counter()
    by_month = Counter()
    conv_stats = []
    all_canonical = []

    conv_dirs = sorted([d for d in inbox.iterdir() if d.is_dir()])
    report.total_conversations = 0
    print(f"Scanning {len(conv_dirs)} conversation folders...")

    for i, conv_dir in enumerate(conv_dirs):
        conv_messages = []
        title = conv_dir.name
        participant_count = 2
        thread_id = conv_dir.name
        target_found_in_conv = False

        for msg_file in sorted(conv_dir.glob("message_*.json")):
            try:
                with open(msg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            title = fix_mojibake(data.get("title", title))
            participant_count = len(data.get("participants", []))
            thread_id = data.get("thread_path", thread_id)

            # check if target participates in this conversation
            participants = [fix_mojibake(p.get("name", "")) for p in data.get("participants", [])]
            if report.target_name in participants:
                target_found_in_conv = True

            for msg in data.get("messages", []):
                canonical = to_canonical(msg, thread_id, msg_file.name, report.target_name)
                conv_messages.append(canonical)

        if not conv_messages or not target_found_in_conv:
            continue

        report.total_conversations += 1

        # reverse to chronological (FB exports newest-first)
        conv_messages.reverse()
        all_canonical.extend(conv_messages)

        # stats
        target_msgs = [m for m in conv_messages if m["is_target"]]
        report.total_messages += len(conv_messages)
        report.target_messages += len(target_msgs)

        if participant_count > 2:
            report.group_chats += 1
        else:
            report.dm_chats += 1

        for m in target_msgs:
            type_counts[m["msg_type"]] += 1
            len_buckets[length_bucket(len(m["text"]))] += 1
            if m["timestamp"]:
                by_month[m["timestamp"][:7]] += 1

        # per-conversation summary
        target_text_msgs = [m for m in target_msgs if m["msg_type"] == "text"]
        avg_len = (
            sum(len(m["text"]) for m in target_text_msgs) / len(target_text_msgs)
            if target_text_msgs
            else 0
        )
        timestamps = [m["timestamp_ms"] for m in conv_messages if m["timestamp_ms"]]
        date_range = ["", ""]
        if timestamps:
            date_range = [
                datetime.fromtimestamp(min(timestamps) / 1000).strftime("%Y-%m-%d"),
                datetime.fromtimestamp(max(timestamps) / 1000).strftime("%Y-%m-%d"),
            ]

        conv_stats.append({
            "title": title,
            "thread_id": thread_id,
            "participant_count": participant_count,
            "total_messages": len(conv_messages),
            "target_messages": len(target_msgs),
            "target_text_messages": len(target_text_msgs),
            "target_avg_length": round(avg_len, 1),
            "date_range": date_range,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(conv_dirs)}...")

    report.type_counts = dict(type_counts)
    report.length_buckets = dict(len_buckets)
    report.messages_by_month = dict(by_month)
    report.conversations = sorted(conv_stats, key=lambda c: c["target_text_messages"], reverse=True)

    return report, all_canonical


def print_report(r: AuditReport):
    label = "you" if r.is_self else "them"

    print("\n" + "=" * 60)
    print("FACEBOOK MESSENGER DATA AUDIT")
    print("=" * 60)

    print(f"\nTarget: {r.target_name} ({'your twin' if r.is_self else 'other person twin'})")
    print(f"Conversations with target: {r.total_conversations} ({r.dm_chats} DMs, {r.group_chats} groups)")
    print(f"Total messages in those threads: {r.total_messages}")
    print(f"Target's messages: {r.target_messages}")

    # type breakdown
    print(f"\n--- {r.target_name}'s Message Types ---")
    for mtype in ["text", "short", "media_only", "link_share", "system", "empty"]:
        count = r.type_counts.get(mtype, 0)
        pct = count / r.target_messages * 100 if r.target_messages else 0
        bar = "█" * int(pct / 2)
        print(f"  {mtype:>12s}: {count:>6d} ({pct:5.1f}%) {bar}")

    usable = r.type_counts.get("text", 0)
    noise = r.target_messages - usable
    if r.target_messages:
        print(f"\n  Usable (text): {usable}  |  Noise: {noise} ({noise/r.target_messages*100:.1f}%)")

    # length distribution
    print(f"\n--- {r.target_name}'s Text Message Length Distribution ---")
    for bucket in ["1-5", "6-20", "21-50", "51-100", "101-300", "300+"]:
        count = r.length_buckets.get(bucket, 0)
        pct = count / r.target_messages * 100 if r.target_messages else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>7s}: {count:>6d} ({pct:5.1f}%) {bar}")

    # activity timeline
    print(f"\n--- Most Active Months ---")
    sorted_months = sorted(r.messages_by_month.items(), key=lambda x: x[1], reverse=True)[:10]
    for month, count in sorted_months:
        bar = "█" * (count // 20)
        print(f"  {month}: {count:>5d} {bar}")

    # top conversations
    print(f"\n--- Top 15 Conversations (by {r.target_name}'s text messages) ---")
    for c in r.conversations[:15]:
        print(
            f"  {c['title'][:28]:<28s}  "
            f"texts:{c['target_text_messages']:>4d}  "
            f"avg:{c['target_avg_length']:>5.0f}ch  "
            f"{'DM' if c['participant_count'] <= 2 else 'GRP':>3s}  "
            f"{c['date_range'][0]}→{c['date_range'][1]}"
        )

    # recommendations
    print(f"\n--- Recommendations ---")
    if not r.is_self:
        print(f"  ℹ Building twin of another person from YOUR data export.")
        print(f"    You only have {label}'s messages from conversations with you.")
        print(f"    This captures how they talk TO YOU, not how they talk in general.")

    if usable < 500:
        print(f"  ⚠ Only {usable} usable text messages. Very thin for personality cloning.")
        if r.is_self:
            print(f"    Add more sources (Telegram, email) to reach 3000+.")
        else:
            print(f"    Consider asking this person to export their own data for richer signal.")
    elif usable < 3000:
        print(f"  ~ {usable} usable messages. Workable for system prompt + few-shot RAG.")
    else:
        print(f"  ✓ {usable} usable text messages. Solid foundation.")

    short_pct = r.type_counts.get("short", 0) / r.target_messages * 100 if r.target_messages else 0
    if short_pct > 40:
        print(f"  ⚠ {short_pct:.0f}% of messages are ≤5 chars. Terse style —")
        print(f"    the system prompt should reflect this, not fight it.")

    if r.group_chats > r.dm_chats:
        print(f"  ⚠ More groups than DMs. Prioritize DM threads for personality signal.")

    print("=" * 60)


def save_outputs(r: AuditReport, canonical_msgs: list, base_dir: str):
    # organize by target name: data/<target_name>/
    safe_name = r.target_name.replace(" ", "_").lower()
    output_dir = os.path.join(base_dir, "data", safe_name)
    os.makedirs(output_dir, exist_ok=True)

    # audit report
    report_path = os.path.join(output_dir, "audit_report.json")
    report_data = {
        "target_name": r.target_name,
        "is_self": r.is_self,
        "total_conversations": r.total_conversations,
        "dm_chats": r.dm_chats,
        "group_chats": r.group_chats,
        "total_messages": r.total_messages,
        "target_messages": r.target_messages,
        "type_counts": r.type_counts,
        "length_buckets": r.length_buckets,
        "messages_by_month": r.messages_by_month,
        "top_conversations": r.conversations[:50],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nAudit report → {report_path}")

    # cleaned JSONL — all messages for context, tagged with is_target
    jsonl_path = os.path.join(output_dir, "cleaned_messages.jsonl")
    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for msg in canonical_msgs:
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
            written += 1
    print(f"Cleaned messages → {jsonl_path} ({written} messages)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit Facebook Messenger data for digital twin pipeline"
    )
    parser.add_argument("inbox", help="Path to messages/inbox folder")
    parser.add_argument(
        "--self",
        help="Your name in the export (skips auto-detection)",
    )
    parser.add_argument(
        "--target",
        help="Name of the person to build a twin of (default: yourself)",
    )
    parser.add_argument(
        "--list-people",
        action="store_true",
        help="List all people found in the inbox and exit",
    )
    args = parser.parse_args()

    inbox = Path(args.inbox)
    if not inbox.is_dir():
        print(f"Error: {args.inbox} is not a directory")
        sys.exit(1)

    if args.list_people:
        print("Scanning all senders (this may take a minute)...")
        people = list_all_people(inbox)
        print(f"\n{'Name':<40s}  {'Messages':>8s}")
        print("-" * 52)
        for name, count in people:
            print(f"  {name:<40s}  {count:>8d}")
        print(f"\nTotal unique senders: {len(people)}")
        print(f"\nTo build a twin, run:")
        print(f'  python audit_facebook.py {args.inbox} --target "Person Name"')
        sys.exit(0)

    report, canonical = run_audit(args.inbox, self_name=getattr(args, 'self', None), target_name=args.target)
    print_report(report)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_outputs(report, canonical, base_dir)

"""Segment normalization and chunk building.

Takes boundary indices + messages, normalizes segment sizes
(merge tiny, split large), and builds chunk dicts for ChromaDB.
"""


def normalize_segments(
    messages: list[dict],
    boundaries: list[int],
    min_size: int = 3,
    max_size: int = 20,
) -> list[list[dict]]:
    """Split messages into segments at boundaries, then normalize sizes."""
    if not messages:
        return []

    cuts = [0] + sorted(boundaries) + [len(messages)]
    raw_segments = []
    for i in range(len(cuts) - 1):
        seg = messages[cuts[i]:cuts[i + 1]]
        if seg:
            raw_segments.append(seg)

    if not raw_segments:
        return []

    merged = []
    carry: list[dict] = []
    for seg in raw_segments:
        combined = carry + seg
        if len(combined) < min_size:
            carry = combined
        else:
            merged.append(combined)
            carry = []
    if carry:
        if merged:
            merged[-1].extend(carry)
        else:
            merged.append(carry)

    final = []
    for seg in merged:
        if len(seg) <= max_size:
            final.append(seg)
        else:
            n_parts = (len(seg) + max_size - 1) // max_size
            part_size = (len(seg) + n_parts - 1) // n_parts
            for i in range(0, len(seg), part_size):
                part = seg[i:i + part_size]
                if part:
                    final.append(part)

    return final


def build_chunks(
    segments: list[list[dict]],
    thread_id: str,
    twin_name: str,
) -> list[dict]:
    """Build chunk dicts from normalized segments."""
    chunks = []
    for i, seg in enumerate(segments):
        if not seg:
            continue

        doc_lines = []
        for m in seg:
            doc_lines.append(f"{m.get('author', '?')}: {m.get('text', '')}")
        document = "\n".join(doc_lines)

        participants = sorted(set(m.get("author", "?") for m in seg))
        twin_msgs = [m for m in seg if m.get("author") == twin_name]

        chunks.append({
            "chunk_id": f"{thread_id}_seg{i}",
            "messages": seg,
            "document": document,
            "metadata": {
                "msg_count": len(seg),
                "participants": participants,
                "time_start": seg[0].get("timestamp", ""),
                "time_end": seg[-1].get("timestamp", ""),
                "twin_msg_ratio": round(len(twin_msgs) / len(seg), 3) if seg else 0.0,
                "thread_id": thread_id,
            },
        })

    return chunks

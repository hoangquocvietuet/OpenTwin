"""
Richness Scorer & Conversation Chunker

Reads cleaned_messages.jsonl and produces:
1. Scored conversation chunks (context → target response pairs)
2. Style fingerprint of the target person
3. Holdout dataset (10% stratified sample for evaluation)

Strategy:
- DMs: straightforward turn-based pairing (gold standard)
- Groups: time-gated extraction (2-min window = likely reply)
- Scoring: length, turn depth, topic richness

Usage:
    python score_and_chunk.py data/hoàng_quốc_việt/cleaned_messages.jsonl
"""

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# =============================================================================
# Load data
# =============================================================================

def load_jsonl(path: str) -> list[dict]:
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def group_by_thread(messages: list[dict]) -> dict[str, list[dict]]:
    threads = {}
    for msg in messages:
        tid = msg["thread_id"]
        if tid not in threads:
            threads[tid] = []
        threads[tid].append(msg)
    # sort each thread by timestamp
    for tid in threads:
        threads[tid].sort(key=lambda m: m.get("timestamp") or "")
    return threads


# =============================================================================
# Chunk extraction
# =============================================================================

# Facebook auto-reply templates and bot messages that look like user text
BOT_PATTERNS = [
    "bằng việc trả lời, bạn sẽ bắt đầu",
    "by replying, you will begin",
    "you are now connected",
    "bạn đã kết nối với",
    "tin nhắn tự động",
    "auto-reply",
]


def is_bot_message(text: str) -> bool:
    text_lower = text.lower()
    return any(p in text_lower for p in BOT_PATTERNS)


def time_gate_for_group(num_authors: int) -> int:
    """Adaptive time gate based on group size.

    Smaller groups = more coherent threads = longer acceptable gaps.
    Larger groups = more noise = tighter window needed.

    3 people:  10 min  (almost like a DM, high coherence)
    5 people:   5 min
    10 people:  2 min
    20+ people: 1 min  (lots of cross-talk, tight gate)
    """
    if num_authors <= 3:
        return 600
    if num_authors <= 5:
        return 300
    if num_authors <= 10:
        return 120
    return 60


@dataclass
class Chunk:
    """A context → response pair for RAG or evaluation."""
    chunk_id: str
    thread_id: str
    thread_title: str = ""
    context: list[dict] = field(default_factory=list)  # preceding messages
    response: dict = field(default_factory=dict)        # target's message
    chunk_type: str = ""   # "dm" or "group"
    score: float = 0.0
    # scoring components
    response_length: int = 0
    context_turns: int = 0
    has_question: bool = False
    time_gap_seconds: float = 0.0


def is_dm_thread(thread_msgs: list[dict]) -> bool:
    """A thread is DM if it has exactly 2 unique authors."""
    authors = set(m["author"] for m in thread_msgs if m["author"])
    return len(authors) <= 2


def extract_dm_chunks(thread_msgs: list[dict], thread_id: str) -> list[Chunk]:
    """Extract turn-based pairs from DM conversations.

    For each target message, grab up to 5 preceding messages as context.
    """
    chunks = []
    for i, msg in enumerate(thread_msgs):
        if not msg.get("is_target"):
            continue
        if msg.get("msg_type") not in ("text", "link_share"):
            continue
        if not msg.get("text", "").strip():
            continue
        if is_bot_message(msg.get("text", "")):
            continue

        # grab preceding context (up to 5 messages)
        context_start = max(0, i - 5)
        context = []
        for c in thread_msgs[context_start:i]:
            if c.get("msg_type") in ("text", "link_share", "short") and c.get("text", "").strip() and not is_bot_message(c.get("text", "")):
                context.append({
                    "author": c["author"],
                    "text": c["text"],
                    "timestamp": c["timestamp"],
                    "is_target": c["is_target"],
                })

        chunks.append(Chunk(
            chunk_id=f"dm_{thread_id}_{i}",
            thread_id=thread_id,
            context=context,
            response={
                "author": msg["author"],
                "text": msg["text"],
                "timestamp": msg["timestamp"],
            },
            chunk_type="dm",
            response_length=len(msg["text"]),
            context_turns=len(context),
        ))

    return chunks


def extract_group_chunks(thread_msgs: list[dict], thread_id: str) -> list[Chunk]:
    """Extract time-gated pairs from group conversations.

    Time gate scales with group size:
    - 3 people: 10 min (high coherence, almost DM-like)
    - 5 people: 5 min
    - 10 people: 2 min
    - 20+ people: 1 min (lots of cross-talk)
    """
    num_authors = len(set(m["author"] for m in thread_msgs if m["author"]))
    gate = time_gate_for_group(num_authors)

    chunks = []
    for i, msg in enumerate(thread_msgs):
        if not msg.get("is_target"):
            continue
        if msg.get("msg_type") not in ("text", "link_share"):
            continue
        if not msg.get("text", "").strip():
            continue
        if is_bot_message(msg.get("text", "")):
            continue
        if i == 0:
            continue

        # check time gap to previous message
        prev = thread_msgs[i - 1]
        try:
            ts_cur = datetime.fromisoformat(msg["timestamp"])
            ts_prev = datetime.fromisoformat(prev["timestamp"])
            gap = abs((ts_cur - ts_prev).total_seconds())
        except (ValueError, TypeError):
            gap = float("inf")

        # grab context — only messages within scaled time gate window
        context = []
        for j in range(i - 1, max(i - 6, -1), -1):
            c = thread_msgs[j]
            try:
                ts_c = datetime.fromisoformat(c["timestamp"])
                c_gap = abs((ts_cur - ts_c).total_seconds())
            except (ValueError, TypeError):
                break
            if c_gap > gate * 3:  # wider window for context
                break
            if c.get("msg_type") in ("text", "link_share", "short") and c.get("text", "").strip():
                context.insert(0, {
                    "author": c["author"],
                    "text": c["text"],
                    "timestamp": c["timestamp"],
                    "is_target": c["is_target"],
                })

        chunks.append(Chunk(
            chunk_id=f"grp_{thread_id}_{i}",
            thread_id=thread_id,
            context=context,
            response={
                "author": msg["author"],
                "text": msg["text"],
                "timestamp": msg["timestamp"],
            },
            chunk_type="group",
            response_length=len(msg["text"]),
            context_turns=len(context),
            time_gap_seconds=gap,
        ))

    return chunks


# =============================================================================
# Scoring
# =============================================================================

def score_chunk(chunk: Chunk) -> float:
    """Score a chunk's quality for training/evaluation use.

    Higher = more valuable for personality cloning.
    """
    score = 0.0

    # response length (longer = more personality signal, up to a point)
    rlen = chunk.response_length
    if rlen <= 5:
        score += 0.1
    elif rlen <= 20:
        score += 0.3
    elif rlen <= 50:
        score += 0.6
    elif rlen <= 100:
        score += 0.8
    else:
        score += 1.0

    # context depth (more turns = richer context for RAG)
    score += min(chunk.context_turns * 0.15, 0.6)

    # has question in context (reply-to-question is gold)
    context_text = " ".join(c["text"] for c in chunk.context)
    if "?" in context_text:
        score += 0.3
        chunk.has_question = True

    # DMs are higher quality than group extractions
    if chunk.chunk_type == "dm":
        score += 0.2

    # penalize group messages with large time gaps (likely not a reply)
    # uses a default mid-range gate for scoring since we don't have group size here
    if chunk.chunk_type == "group" and chunk.time_gap_seconds > 120:
        score -= 0.3

    # penalize no context (standalone message, hard to evaluate)
    if chunk.context_turns == 0:
        score -= 0.3

    chunk.score = round(max(0.0, min(score, 2.0)), 2)
    return chunk.score


# =============================================================================
# Style fingerprint
# =============================================================================

@dataclass
class StyleFingerprint:
    total_messages: int = 0
    avg_length: float = 0.0
    median_length: int = 0
    length_distribution: dict = field(default_factory=dict)
    # punctuation
    ends_with_period_pct: float = 0.0
    ends_with_emoji_pct: float = 0.0
    uses_ellipsis_pct: float = 0.0
    uses_exclamation_pct: float = 0.0
    question_mark_pct: float = 0.0
    # casing
    all_lowercase_pct: float = 0.0
    # vocabulary
    top_words: list = field(default_factory=list)
    top_emojis: list = field(default_factory=list)
    # cadence
    avg_words_per_msg: float = 0.0


def extract_emojis(text: str) -> list[str]:
    import re
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U0001f900-\U0001f9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.findall(text)


def build_fingerprint(messages: list[dict]) -> StyleFingerprint:
    """Build a structural style fingerprint from target's messages."""
    target_texts = [
        m["text"] for m in messages
        if m.get("is_target") and m.get("msg_type") == "text" and m.get("text", "").strip()
        and not is_bot_message(m["text"])
    ]

    if not target_texts:
        return StyleFingerprint()

    fp = StyleFingerprint()
    fp.total_messages = len(target_texts)

    lengths = [len(t) for t in target_texts]
    lengths.sort()
    fp.avg_length = round(sum(lengths) / len(lengths), 1)
    fp.median_length = lengths[len(lengths) // 2]

    # length distribution
    buckets = Counter()
    for l in lengths:
        if l <= 5: buckets["1-5"] += 1
        elif l <= 20: buckets["6-20"] += 1
        elif l <= 50: buckets["21-50"] += 1
        elif l <= 100: buckets["51-100"] += 1
        elif l <= 300: buckets["101-300"] += 1
        else: buckets["300+"] += 1
    fp.length_distribution = dict(buckets)

    # punctuation patterns
    n = len(target_texts)
    fp.ends_with_period_pct = round(sum(1 for t in target_texts if t.rstrip().endswith(".")) / n * 100, 1)
    fp.uses_exclamation_pct = round(sum(1 for t in target_texts if "!" in t) / n * 100, 1)
    fp.question_mark_pct = round(sum(1 for t in target_texts if "?" in t) / n * 100, 1)
    fp.uses_ellipsis_pct = round(sum(1 for t in target_texts if "..." in t) / n * 100, 1)
    fp.all_lowercase_pct = round(sum(1 for t in target_texts if t == t.lower()) / n * 100, 1)

    # emojis
    emoji_counts = Counter()
    emoji_msg_count = 0
    for t in target_texts:
        emojis = extract_emojis(t)
        if emojis:
            emoji_msg_count += 1
            emoji_counts.update(emojis)
    fp.ends_with_emoji_pct = round(emoji_msg_count / n * 100, 1)
    fp.top_emojis = emoji_counts.most_common(10)

    # vocabulary
    word_counts = Counter()
    total_words = 0
    for t in target_texts:
        words = t.lower().split()
        total_words += len(words)
        word_counts.update(w for w in words if len(w) > 2)
    fp.top_words = word_counts.most_common(20)
    fp.avg_words_per_msg = round(total_words / n, 1)

    return fp


# =============================================================================
# Holdout sampling
# =============================================================================

def stratified_holdout(chunks: list[Chunk], holdout_pct: float = 0.10, seed: int = 42) -> tuple[list[Chunk], list[Chunk]]:
    """Split chunks into train and holdout sets, stratified by type and score.

    Ensures holdout covers both DM and group, high and low quality.
    """
    random.seed(seed)

    dm_chunks = [c for c in chunks if c.chunk_type == "dm"]
    grp_chunks = [c for c in chunks if c.chunk_type == "group"]

    def sample_stratified(chunk_list):
        n_holdout = max(1, int(len(chunk_list) * holdout_pct))
        # sort by score, take evenly from quartiles
        sorted_chunks = sorted(chunk_list, key=lambda c: c.score)
        step = max(1, len(sorted_chunks) // n_holdout)
        holdout = []
        for i in range(0, len(sorted_chunks), step):
            if len(holdout) < n_holdout:
                holdout.append(sorted_chunks[i])
        return holdout

    dm_holdout = sample_stratified(dm_chunks) if dm_chunks else []
    grp_holdout = sample_stratified(grp_chunks) if grp_chunks else []

    holdout_ids = set(c.chunk_id for c in dm_holdout + grp_holdout)
    train = [c for c in chunks if c.chunk_id not in holdout_ids]
    holdout = dm_holdout + grp_holdout

    return train, holdout


# =============================================================================
# Output
# =============================================================================

def chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "thread_id": chunk.thread_id,
        "chunk_type": chunk.chunk_type,
        "score": chunk.score,
        "context": chunk.context,
        "response": chunk.response,
        "response_length": chunk.response_length,
        "context_turns": chunk.context_turns,
        "has_question": chunk.has_question,
        "time_gap_seconds": chunk.time_gap_seconds,
    }


def print_fingerprint(fp: StyleFingerprint):
    print("\n" + "=" * 60)
    print("STYLE FINGERPRINT")
    print("=" * 60)
    print(f"\nMessages analyzed: {fp.total_messages}")
    print(f"Avg length: {fp.avg_length} chars | Median: {fp.median_length} chars")
    print(f"Avg words/msg: {fp.avg_words_per_msg}")

    print(f"\n--- Length Distribution ---")
    for bucket in ["1-5", "6-20", "21-50", "51-100", "101-300", "300+"]:
        count = fp.length_distribution.get(bucket, 0)
        pct = count / fp.total_messages * 100 if fp.total_messages else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>7s}: {count:>4d} ({pct:5.1f}%) {bar}")

    print(f"\n--- Punctuation & Style ---")
    print(f"  All lowercase:  {fp.all_lowercase_pct}%")
    print(f"  Ends with '.':  {fp.ends_with_period_pct}%")
    print(f"  Uses '!':       {fp.uses_exclamation_pct}%")
    print(f"  Uses '?':       {fp.question_mark_pct}%")
    print(f"  Uses '...':     {fp.uses_ellipsis_pct}%")
    print(f"  Has emoji:      {fp.ends_with_emoji_pct}%")

    if fp.top_emojis:
        print(f"\n--- Top Emojis ---")
        for emoji, count in fp.top_emojis:
            print(f"  {emoji} × {count}")

    print(f"\n--- Top Words (>2 chars) ---")
    for word, count in fp.top_words:
        print(f"  {word:<20s} {count:>4d}")

    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_and_chunk.py data/<target>/cleaned_messages.jsonl")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    output_dir = str(Path(jsonl_path).parent)

    print(f"Loading {jsonl_path}...")
    messages = load_jsonl(jsonl_path)
    print(f"  {len(messages)} messages loaded")

    # group by thread
    threads = group_by_thread(messages)
    print(f"  {len(threads)} threads")

    # extract chunks
    all_chunks = []
    dm_count = 0
    grp_count = 0

    for tid, thread_msgs in threads.items():
        if is_dm_thread(thread_msgs):
            chunks = extract_dm_chunks(thread_msgs, tid)
            dm_count += 1
        else:
            chunks = extract_group_chunks(thread_msgs, tid)
            grp_count += 1
        all_chunks.extend(chunks)

    print(f"\nExtracted {len(all_chunks)} chunks ({dm_count} DM threads, {grp_count} group threads)")

    # score
    for chunk in all_chunks:
        score_chunk(chunk)

    scored = sorted(all_chunks, key=lambda c: c.score, reverse=True)

    # stats
    scores = [c.score for c in scored]
    avg_score = sum(scores) / len(scores) if scores else 0
    high_quality = [c for c in scored if c.score >= 0.8]
    print(f"Avg score: {avg_score:.2f} | High quality (>=0.8): {len(high_quality)}")

    # score distribution
    print(f"\n--- Score Distribution ---")
    for threshold in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]:
        count = sum(1 for s in scores if s >= threshold)
        print(f"  >= {threshold:.1f}: {count:>4d}")

    # holdout split
    train, holdout = stratified_holdout(scored)
    print(f"\nTrain: {len(train)} | Holdout: {len(holdout)}")

    # style fingerprint
    fp = build_fingerprint(messages)
    print_fingerprint(fp)

    # save outputs
    train_path = f"{output_dir}/train_chunks.jsonl"
    holdout_path = f"{output_dir}/holdout_chunks.jsonl"
    fingerprint_path = f"{output_dir}/style_fingerprint.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for chunk in train:
            f.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")
    print(f"\nTrain chunks → {train_path}")

    with open(holdout_path, "w", encoding="utf-8") as f:
        for chunk in holdout:
            f.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")
    print(f"Holdout chunks → {holdout_path}")

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
    with open(fingerprint_path, "w", encoding="utf-8") as f:
        json.dump(fp_dict, f, indent=2, ensure_ascii=False)
    print(f"Style fingerprint → {fingerprint_path}")

    # sample top chunks
    print(f"\n--- Top 5 Chunks (highest score) ---")
    for chunk in scored[:5]:
        ctx_preview = " | ".join(c["text"][:40] for c in chunk.context[-2:])
        print(f"  [{chunk.score:.2f}] ({chunk.chunk_type}) ctx: {ctx_preview}")
        print(f"         → {chunk.response['text'][:80]}")
        print()


if __name__ == "__main__":
    main()

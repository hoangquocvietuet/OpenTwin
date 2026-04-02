# AI Pipeline Design: Multi-Agent LangGraph Pipeline with Rich Metadata

## Problem

The current system has a single RAG pipeline: cosine similarity search → stuff chunks into prompt → one LLM call. This causes:

1. **No intent detection** — same retrieval for "tối nay ăn j" and a 500-word news article
2. **Content-only matching** — cosine on embeddings doesn't consider tone, formality, emotion, or persona
3. **Content leaks into rewrite mode** — retrieved chunks about food cause the LLM to answer about food instead of rephrasing
4. **Fixed chunking** — single messages lack context, large chunks add noise
5. **No quality gate** — wrong responses (answered instead of rewriting, wrong tone) go straight to the user

## Solution Overview

A 4-agent sequential pipeline built on **LangGraph**, with a rich **metadata enrichment system** at import time and **dynamic chunking** based on conversation boundaries.

**Key principle:** Content follows the input, tone and persona follow old memories.

## Architecture

### Pipeline Flow

```
User input
  → [Intent Agent]       classify intent, tone, detect if context needed
  → [Context Agent]      (conditional) fetch URL / clipboard if needed
  → [Retriever Agent]    hybrid retrieval: tone-match + content-match
  → [Responder Agent]    generate response using prompts + retrieved context
  → [Critic Agent]       review: tone match, style match, no hallucination
       ↓ fail (max 2 retries)
       → back to Responder with feedback
       ↓ pass
  → Final response
```

### Pipeline State

Shared state that flows through the graph. All agents read/write to this.

```python
@dataclass
class PipelineState:
    # Input
    raw_input: str                    # what the user typed
    mode: str                         # "answer" | "rewrite"

    # Intent Agent output
    intent: str | None                # "casual_chat", "question", "rewrite_announcement", ...
    tone: str | None                  # "formal", "casual", "banter", "sarcastic", ...
    needs_context: bool               # True if input is a command, not full content
    context_source: str | None        # "url", "clipboard", None
    context_url: str | None           # extracted URL if present

    # Context Agent output (optional step)
    resolved_content: str | None      # the actual text to work with

    # Retriever Agent output
    tone_chunks: list[dict]           # chunks matched by tone/persona
    content_chunks: list[dict]        # chunks matched by content similarity

    # Responder Agent output
    draft_response: str | None

    # Critic Agent output
    approved: bool
    critic_feedback: str | None
    retry_count: int
```

### LangGraph Definition

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(PipelineState)
graph.add_node("intent", intent_agent)
graph.add_node("context", context_agent)
graph.add_node("retriever", retriever_agent)
graph.add_node("responder", responder_agent)
graph.add_node("critic", critic_agent)

graph.set_entry_point("intent")
graph.add_conditional_edges("intent", route_after_intent, {
    "needs_context": "context",
    "has_content": "retriever",
})
graph.add_edge("context", "retriever")
graph.add_edge("retriever", "responder")
graph.add_edge("responder", "critic")
graph.add_conditional_edges("critic", should_retry, {
    "retry": "responder",
    "done": END,
})
```

## Agent Details

### 1. Intent Agent

Classifies the input and decides routing. Uses the configured classifier model.

**Input:** `raw_input`, `mode`
**Output:** `intent`, `tone`, `needs_context`, `context_source`, `context_url`, `resolved_content`

Behavior:
- If input is long text → `needs_context=False`, `resolved_content=raw_input`
- If input is short + contains URL → `needs_context=True`, `context_source="url"`
- If input is short + looks like a command → `needs_context=True`, `context_source="clipboard"`

LLM prompt returns structured JSON:
```json
{
  "intent": "rewrite_article",
  "tone": "formal_news",
  "needs_context": false,
  "context_source": null
}
```

### 2. Context Agent

Conditional — only runs if `needs_context=True`. Fetches the actual content.

**Input:** `context_source`, `context_url`, `raw_input`
**Output:** `resolved_content`

Sources:
- `"url"`: fetch article from URL, extract text
- `"clipboard"`: return instruction for client to provide clipboard (future: browser extension)
- Future extensible to other sources

After fetching, the Context Agent internally calls the same intent classification LLM prompt on the resolved content to determine tone (since tone couldn't be determined from just a URL). It updates `tone` and `intent` on the state.

### 3. Retriever Agent

Hybrid retrieval: tone-match for style, content-match for personal context.

**Input:** `resolved_content`, `intent`, `tone`
**Output:** `tone_chunks` (top 3), `content_chunks` (top 2)

**Tone retrieval:**
```python
collection.query(
    query_texts=[resolved_content],
    n_results=10,
    where={
        "$and": [
            {"tone": {"$in": [detected_tone, ...similar_tones]}},  # similarity map in config
            {"formality": {"$lte": threshold}},
            {"twin_msg_ratio": {"$gte": 0.3}},
        ]
    },
)
# Re-rank by: persona match, emotion alignment, recency
# Return top 3
```

**Content retrieval:**
```python
collection.query(
    query_texts=[resolved_content],
    n_results=5,
    where={"twin_msg_ratio": {"$gte": 0.2}},
)
# Filter: only keep if cosine similarity > 0.6
# If nothing passes → empty (no noise is better than bad matches)
# Return top 2
```

### 4. Responder Agent

Generates the response. Uses the main configured LLM model.

**Input:** `resolved_content`, `intent`, `tone`, `tone_chunks`, `content_chunks`, system/rewrite prompt, `critic_feedback` (if retry)
**Output:** `draft_response`

Prompt structure:
- System prompt (answer_prompt or rewrite_prompt from `prompt.py`)
- Tone chunks labeled: "This is how you sound in this register: ..."
- Content chunks labeled: "This is what you've said about similar topics: ..."
- Conversation history (answer mode only)
- Critic feedback (if retrying): "Previous attempt was rejected because: ..."
- User message

### 5. Critic Agent

Reviews the response before sending. Uses the configured classifier model.

**Input:** `raw_input`, `resolved_content`, `draft_response`, `intent`, `tone`, `mode`, `tone_chunks`
**Output:** `approved` (bool), `critic_feedback` (str)

Checks:
- **Style match:** Does it sound like the twin? Compare against tone_chunks.
- **Tone match:** Casual input → casual output, formal → formal.
- **Mode compliance (rewrite):** Did it rewrite, not answer? Statements stay statements, questions stay questions.
- **Mode compliance (answer):** On-topic? No hallucination?
- **Length:** Appropriate for the input energy? Not an essay for a short question.

If fail → `critic_feedback` describes what's wrong. Sent back to Responder. Max 2 retries.

## Dynamic Chunking System

Replaces the current fixed chunking in `score_and_chunk.py`.

### Pass 1: Boundary Detection

Uses a configurable small LLM to scan conversations with a sliding window. For each message boundary, the LLM decides: "same interaction or new one?"

Signals considered:
- **Topic shift** — "food" → "work"
- **Mood shift** — playful → angry
- **Participant change** — new person enters conversation
- **Time gap** — as a hint only, not a hard rule. 3hr gap on the same topic stays together.

Output: list of boundary indices splitting the conversation into segments.

### Pass 2: Size Normalization

- Segments > 20 messages → re-split at sub-boundaries (ask LLM for finer cuts)
- Segments < 3 messages → merge with the most similar neighbor
- Target: **3-20 messages per chunk**, each a coherent interaction arc

### Chunk Structure

```python
{
    "chunk_id": "dm_thread123_seg5",
    "messages": [...],               # raw messages in order
    "document": "...",               # formatted for embedding
    "context_summary": "...",        # LLM-generated: what's happening in this chunk
    "metadata": {
        "msg_count": 8,
        "participants": ["Hoang Quoc Viet", "Nguyen Tuan Tai"],
        "time_start": "2025-08-01T10:00:00",
        "time_end": "2025-08-01T10:15:00",
        "twin_msg_ratio": 0.6,
        # ... enrichment metadata from analyzers
    }
}
```

## ChromaDB Collection Settings

All collections MUST be created with cosine distance (not the ChromaDB default of L2/Euclidean). This is critical — L2 distances are unbounded, making `1 - distance` produce negative similarity scores.

```python
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"},  # REQUIRED — never omit
    embedding_function=embedding_fn,
)
```

The `importer.py`, `backfill.py`, `rechunk.py`, and `pipeline/agents/retriever.py` must all enforce this. Existing collections created without cosine require re-import (`rechunk` handles this).

## Metadata Enrichment System

### Analyzer Registry

A registry of analyzer functions that enrich chunks at import time. Each analyzer produces metadata fields. New analyzers can be added over time.

```python
@dataclass
class AnalyzerInput:
    chunk: dict                    # the chunk being analyzed
    prev_chunk: dict | None        # previous chunk in same thread
    next_chunk: dict | None        # next chunk in same thread
    thread_summary: str | None     # running summary of the thread so far

ANALYZER_REGISTRY = {
    "context_v1": {
        "fn": analyze_context,
        # → {"context_summary": "friends planning dinner, casual tone",
        #    "interaction_type": "planning", "relationship": "close_friends"}
        "requires_llm": True,
        "run_order": 0,           # runs first, others can use its output
    },
    "tone_v1": {
        "fn": analyze_tone,
        # → {"tone": "casual_banter", "formality": 0.2, "energy": "relaxed"}
        "requires_llm": True,
        "run_order": 1,
    },
    "emotion_v1": {
        "fn": analyze_emotion,
        # → {"emotion": "playful", "sentiment": 0.7,
        #    "conflict": false, "sarcasm": false}
        "requires_llm": True,
        "run_order": 1,
    },
    "persona_v1": {
        "fn": analyze_persona,
        # → {"twin_role": "initiator", "register": "informal_close",
        #    "relationship_to_others": "friend_banter"}
        "requires_llm": True,
        "run_order": 1,
    },
    "stats_v1": {
        "fn": analyze_stats,
        # → {"avg_msg_len": 15, "emoji_count": 2, "language": "vi",
        #    "question_ratio": 0.3, "twin_msg_ratio": 0.6}
        "requires_llm": False,
        "run_order": 0,
    },
}
```

### Context-Aware Analysis

Analyzers receive the full chunk (multiple messages in sequence) plus neighboring chunks for broader context. `run_order=0` analyzers run first — their output (e.g. `context_summary`) is available to `run_order=1` analyzers, enabling context-aware classification.

### Backfill & Rechunk

Each chunk stores `_analyzers_applied: {"tone_v1": 1, "stats_v1": 1, ...}` in metadata.

**Backfill** (`python -m app.backfill`):
- Compares each chunk's `_analyzers_applied` against the registry
- Runs only missing or outdated analyzers
- `--analyzer tone_v1` to run a specific one

**Rechunk** (`python -m app.rechunk`):
- Re-runs boundary detection on raw messages
- Re-creates chunks with new boundaries
- Re-runs all analyzers on new chunks

## Configuration

All models are configurable. Each pipeline role has its own config with fallback to the main LLM:

```env
# Main LLM (Responder Agent)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:8b
LLM_API_KEY=ollama

# Classifier model (Intent Agent, Critic Agent) — falls back to LLM_*
CLASSIFIER_BASE_URL=http://localhost:11434/v1
CLASSIFIER_MODEL=llama3.2:3b
CLASSIFIER_API_KEY=ollama

# Analyzer model (import-time enrichment, chunking) — falls back to LLM_*
ANALYZER_BASE_URL=http://localhost:11434/v1
ANALYZER_MODEL=llama3.2:3b
ANALYZER_API_KEY=ollama
```

## File Structure

```
app/
  pipeline/                  # NEW — LangGraph pipeline
    __init__.py
    state.py                 # PipelineState dataclass
    graph.py                 # LangGraph graph definition + compilation
    agents/
      __init__.py
      intent.py              # Intent Agent
      context.py             # Context Agent (URL fetch, etc.)
      retriever.py           # Retriever Agent (tone + content queries)
      responder.py           # Responder Agent (generates response)
      critic.py              # Critic Agent (review + retry logic)
  analyzers/                 # NEW — metadata enrichment system
    __init__.py
    registry.py              # Analyzer registry + runner
    tone.py                  # tone_v1 analyzer
    emotion.py               # emotion_v1 analyzer
    persona.py               # persona_v1 analyzer
    context.py               # context_v1 analyzer (chunk summarization)
    stats.py                 # stats_v1 analyzer (heuristic, no LLM)
  chunking/                  # NEW — replaces score_and_chunk.py
    __init__.py
    boundary.py              # LLM-based boundary detection
    normalizer.py            # merge small, split large segments
  backfill.py                # NEW — CLI: python -m app.backfill
  rechunk.py                 # NEW — CLI: python -m app.rechunk

  # MODIFIED
  chat_service.py            # Calls pipeline.graph.run() instead of inline logic
  config.py                  # Add CLASSIFIER_MODEL, ANALYZER_MODEL configs
  importer.py                # Calls new chunking + analyzers instead of score_and_chunk.py
  prompt.py                  # Kept — prompts used by Responder Agent
```

## Migration Path

The pipeline is opt-in. `chat_service.chat()` detects whether enriched metadata exists:

```python
def chat(content, collection, ...):
    if _has_enriched_metadata(collection):
        return pipeline_chat(content, collection, ...)  # new pipeline
    else:
        return _legacy_chat(content, collection, ...)   # current code
```

1. Ship pipeline code without breaking existing users
2. Users re-import or run `backfill` + `rechunk` to get enriched data
3. Pipeline activates automatically once metadata is present

## Dependencies

New packages:
- `langgraph` — pipeline orchestration
- `langchain-core` — required by LangGraph for message types

No other new dependencies. ChromaDB, OpenAI client, FastAPI, Gradio all stay.

## Tone Similarity Map

Configurable mapping used by the Retriever Agent to expand tone queries. Stored in `app/pipeline/tone_map.py`:

```python
TONE_SIMILARITY = {
    "casual_banter": ["casual", "playful", "friendly"],
    "casual": ["casual_banter", "friendly", "relaxed"],
    "formal_news": ["informational", "serious", "formal"],
    "formal": ["formal_news", "serious", "informational"],
    "sarcastic": ["playful", "casual_banter", "humorous"],
    "emotional": ["serious", "vulnerable", "personal"],
    "angry": ["confrontational", "frustrated", "serious"],
    # extensible — add new tones as analyzers evolve
}
```

When the Intent Agent detects `tone="casual_banter"`, the Retriever queries for chunks with tone in `["casual_banter", "casual", "playful", "friendly"]`.

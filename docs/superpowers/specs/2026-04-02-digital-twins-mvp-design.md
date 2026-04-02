# Digital Twins MVP — Design Spec

> **Approach:** Chat-first minimal (Approach B from CEO review).
> Prove the core value — does RAG + system prompt produce a convincing twin? — before building the platform.
> Full platform features (auth, profiles, privacy) are Phase 2.

## Overview

A privacy-first, locally-deployable platform that creates AI digital twins from a person's chat history. Users import their social media data (Facebook Messenger, later Telegram/Twitter), and the system builds a conversational AI that reproduces their tone, vocabulary, and personality.

## Architecture

**Monolith in a single Docker container.** FastAPI handles logic, Gradio provides the web UI, ChromaDB runs embedded for vector search, SQLite stores chat history. LLM inference is external (Ollama, OpenAI, OpenRouter — any OpenAI-compatible API).

```
                          ┌─────────────────────────────┐
                          │     Docker Container        │
                          │         :7860                │
                          │                             │
  User ──────────────────▶│  Gradio UI (2 tabs)         │
                          │    ├── Chat tab              │
                          │    └── Import tab            │
                          │         │                    │
                          │         ▼                    │
                          │  FastAPI                     │
                          │    ├── POST /api/chat        │
                          │    ├── POST /api/import      │
                          │    └── GET  /api/export      │
                          │         │         │          │
                          │    ┌────┘         └────┐     │
                          │    ▼                   ▼     │
                          │  ChromaDB           SQLite   │
                          │  (embedded)         (chat    │
                          │  (vectors)          history) │
                          └─────────┬───────────────────┘
                                    │
                                    ▼
                          External LLM (Ollama / cloud)
                          POST /v1/chat/completions
```

LLM is NOT bundled. Users run Ollama separately or point to a cloud API.

No auth, no multi-user, no profiles in MVP. Single user, single twin.

## Project Structure

```
digital-twins/
├── app/
│   ├── __init__.py
│   ├── main.py            ← FastAPI app + Gradio mount
│   ├── config.py          ← .env loading (LLM URL, model, paths)
│   ├── database.py        ← SQLite setup (chat history)
│   ├── embedder.py        ← ChromaDB ingestion from JSONL
│   ├── retrieval.py       ← ChromaDB query + context assembly
│   ├── prompt.py          ← system prompt builder from fingerprint
│   ├── chat.py            ← /api/chat endpoint (streaming, OpenAI-compatible)
│   ├── importer.py        ← /api/import endpoint (wraps existing scripts)
│   └── ui.py              ← Gradio chat + import tabs
├── audit_facebook.py      ← existing, imported as module
├── score_and_chunk.py     ← existing, imported as module
├── data/                  ← ChromaDB collections, fingerprints
├── db/                    ← SQLite file
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── ROADMAP.md
```

Existing scripts are imported directly, not refactored. Refactor when adding second data source.

## Data Model (SQLite — chat history only)

### ChatMessage
| Field | Type | Description |
|-------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| twin_slug | str | Which twin (ChromaDB collection name) |
| role | str | "user" or "assistant" |
| content | text | Message text |
| retrieval_metadata | JSON | Chunks retrieved, similarity scores |
| tokens_used | int | For tracking usage |
| created_at | datetime | |

No User or Profile tables in MVP. Single user, single twin.

## Chat Pipeline

```
User message
    │
    ▼
[1. Validate] → reject empty, truncate >10K chars
    │
    ▼
[2. Retrieve] → query twin's ChromaDB collection
    │            - embed user message
    │            - top 5 similar chunks
    │            - if 0 results → return fallback message
    │
    ▼
[3. Build prompt]
    │   - System prompt (auto-generated from fingerprint)
    │   - Few-shot examples (top 3 retrieved chunks)
    │   - Recent chat history (last 10 messages from SQLite)
    │   - User's current message
    │
    ▼
[4. LLM call] → POST {LLM_BASE_URL}/chat/completions
    │            - stream=True (SSE)
    │            - 30-second timeout
    │
    ▼
[5. Stream + save] → stream tokens to Gradio UI
    │                 save complete response to SQLite
    │                 include retrieval metadata (chunk count, avg similarity)
    │
    ▼
[6. Quality indicator] → display below response:
                         "Matched N chunks (avg similarity: X.XX)"
```

### Error handling (all chat path errors)

| Error | Rescue Action | User Sees |
|-------|--------------|-----------|
| LLM unreachable | Catch ConnectionError | "Could not reach the LLM. Check that Ollama is running." |
| LLM timeout (30s) | Catch TimeoutError | "LLM took too long to respond. Try again." |
| LLM rate limit (429) | Catch + show message | "Rate limited. Wait a moment and try again." |
| LLM empty response | Detect empty stream | "No response generated. Try rephrasing." |
| LLM malformed JSON | Catch JSONDecodeError | "Unexpected response format from LLM." |
| ChromaDB not initialized | Check on startup | "No twin data found. Import data first." |
| 0 chunks retrieved | Detect empty results | "I don't have enough context to answer that authentically." |
| SQLite write failure | Catch OperationalError, log | Response still shown, history not saved (warning) |
| Empty user message | Validate before processing | "Please enter a message." |
| Message >10K chars | Truncate to 10K | Silently truncated |

### System Prompt Generation

Auto-generated from `style_fingerprint.json`. Example:

```
You are {twin_name}. Respond as this person would in a casual Vietnamese chat.

STRICT RULES:
1. Write mostly in lowercase ({all_lowercase_pct}% of the time)
2. Keep responses short. Average: {avg_length} chars, median: {median_length} chars
3. Almost never use periods at end of messages (only {ends_with_period_pct}% of the time)
4. Almost never use emojis (only {has_emoji_pct}% of the time). If you do, only 🤣
5. Use Vietnamese particles naturally: nhé, nha, ạ, bác based on context
6. Average {avg_words_per_msg} words per message. Do not write essays.
7. Your most-used words: {top_words}

You are NOT an AI assistant. Do not offer help. Do not be formal. Do not capitalize.
Respond exactly as this person would in a real chat conversation.
```

## Import Pipeline

### Data flow:

```
Upload zip ──▶ Validate (zip? <500MB? no zip slip?) ──▶ Unzip to temp dir
    ──▶ Detect inbox/ folder ──▶ Run audit (audit_facebook.py)
    ──▶ Show audit report in UI ──▶ User clicks "Build Twin"
    ──▶ Run chunker (score_and_chunk.py) ──▶ Generate fingerprint
    ──▶ Check if collection exists (offer overwrite)
    ──▶ Embed chunks into ChromaDB ──▶ Save fingerprint
    ──▶ Auto-switch to Chat tab with "Twin ready!" message
```

### Import error handling:

| Error | Rescue Action | User Sees |
|-------|--------------|-----------|
| Not a zip file | Validate extension + magic bytes | "Please upload a .zip file" |
| Zip > 500MB | Check size before processing | "File too large (max 500MB)" |
| Zip slip (path traversal) | Validate extracted paths | "Invalid zip file" |
| No inbox/ folder found | Check after unzip | "No Facebook messages found in this export" |
| 0 conversations parsed | Check after audit | "No conversations found" |
| Invalid JSON in export | Skip bad files, continue | Warning in audit report |
| ChromaDB embedding failure | Catch and report | "Failed to build twin. Check logs." |
| Collection already exists | Detect before embed | "Twin already exists. Overwrite?" |

### Chunk metadata in ChromaDB:
```json
{
  "chunk_id": "dm_mac_42",
  "chunk_type": "dm",
  "score": 1.7,
  "source_thread": "Mac",
  "timestamp": "2025-08-05T14:30:00",
  "context_turns": 3,
  "response_length": 42
}
```

## Gradio UI

Two tabs:

### Chat Tab (default)
- Chat interface with streaming (Gradio Chatbot component)
- Quality indicator below each response (chunk count + avg similarity)
- Message input + send button
- Export button (download conversation as Markdown/JSON)
- Chat history loaded from SQLite on page open

### Import Tab
- File upload (zip)
- Target name (auto-detect or manual input)
- Audit report display (message counts, quality stats)
- "Build Twin" button
- Progress indicator (parsing... chunking... embedding... N/865)
- Auto-switch to Chat tab on completion

### Interaction States

| Feature | Loading | Empty | Error | Success |
|---------|---------|-------|-------|---------|
| Chat | Streaming tokens | "Import data to start chatting" | Friendly error message | Response + quality indicator |
| Import | Progress bar with stage labels | Upload prompt | Specific error message | "Twin ready!" + auto-switch to Chat |
| Export | "Preparing download..." | N/A | N/A | File download starts |

## Docker Deployment

Single container. LLM is external.

```yaml
# docker-compose.yml
services:
  digital-twins:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./db:/app/db
    env_file: .env
```

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data /app/db
EXPOSE 7860
CMD ["python", "-m", "app.main"]
```

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=llama3.1:8b
TWIN_NAME=auto
CHROMADB_PATH=./data/chromadb
SQLITE_PATH=./db/chat_history.db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Users run: `docker compose up` + `ollama serve` (if using Ollama locally).

**Note:** Document which embedding model is used (`EMBEDDING_MODEL` in .env). Changing it later requires re-embedding all chunks.

## Dependencies

```
fastapi
uvicorn
gradio
sqlalchemy
chromadb
openai
python-multipart
python-dotenv
sentence-transformers
```

## NOT in Scope (Phase 2+)

| Feature | Why deferred | Phase |
|---------|-------------|-------|
| Auth / multi-user | Not needed for single-user MVP | 2 |
| Profiles / multi-twin | Requires auth first | 2 |
| Privacy controls / public mode | Requires auth + multi-user | 2 |
| Settings UI | Hardcode in .env for MVP | 2 |
| Import wizard mode | Quick mode is sufficient to prove value | 3 |
| Telegram/Twitter parsers | Facebook-only for MVP | 3 |
| Evaluation endpoint | Build after twin is working | 4 |
| CLI | API exists, CLI wraps it later | 5 |

## Minimum Tests

| Test | Type | What it verifies |
|------|------|-----------------|
| Upload test zip → chunks in ChromaDB | Integration | Full import pipeline |
| prompt.py builds correct system prompt | Unit | Fingerprint → prompt generation |
| retrieval.py returns ranked chunks | Unit | ChromaDB query + scoring |
| /api/chat returns streaming response | E2E (mock LLM) | Full chat pipeline |

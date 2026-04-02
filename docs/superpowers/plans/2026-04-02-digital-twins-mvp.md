# Digital Twins MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a chat-first MVP that imports Facebook Messenger data and creates a conversational AI twin using RAG + style fingerprinting.

**Architecture:** Single FastAPI app with Gradio UI mounted, ChromaDB embedded for vector search, SQLite for chat history. LLM is external (Ollama or any OpenAI-compatible API). Everything runs in one Docker container on port 7860. Chat logic lives in a shared service layer (`app/chat_service.py`) consumed by both the API endpoint and the Gradio UI — no duplication.

**Tech Stack:** Python 3.13, FastAPI, Gradio, ChromaDB, SQLite (via SQLAlchemy), OpenAI client (for LLM calls), sentence-transformers (for embeddings), python-dotenv

---

## File Structure

```
digital-twins/
├── app/
│   ├── __init__.py          ← empty
│   ├── main.py              ← FastAPI app + Gradio mount, uvicorn entrypoint
│   ├── config.py            ← .env loading, Settings dataclass
│   ├── database.py          ← SQLAlchemy engine + ChatMessage model + session factory
│   ├── embedder.py          ← load train_chunks.jsonl → ChromaDB collection
│   ├── retrieval.py         ← query ChromaDB, return ranked chunks
│   ├── prompt.py            ← build system prompt from style_fingerprint.json
│   ├── chat_service.py      ← shared chat logic (retrieve → prompt → LLM → save)
│   ├── chat.py              ← /api/chat + /api/export endpoints (thin wrappers)
│   ├── importer.py          ← /api/import endpoint (zip → audit → chunk → embed)
│   └── ui.py                ← Gradio Blocks (Chat tab + Import tab)
├── audit_facebook.py        ← existing (imported as module by importer.py)
├── score_and_chunk.py       ← existing (imported as module by importer.py)
├── tests/
│   ├── __init__.py
│   ├── conftest.py          ← shared fixtures (tmp dirs, sample data, test DB)
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_prompt.py
│   ├── test_embedder.py
│   ├── test_retrieval.py
│   ├── test_chat_service.py
│   ├── test_chat.py
│   ├── test_importer.py
│   └── test_ui.py
├── data/                    ← ChromaDB collections, fingerprints (gitignored)
├── db/                      ← SQLite file (gitignored)
├── .env.example
├── .env                     ← copied from .env.example during setup
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**Path convention:** `DATA_DIR` (default `./data`) is the directory that contains twin subdirectories directly (e.g., `./data/hoang_quoc_viet/style_fingerprint.json`). All code — main.py, importer.py, embedder.py — uses this meaning consistently.

---

### Task 1: Project Scaffolding + Config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.env`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write requirements.txt**

```
fastapi==0.115.12
uvicorn[standard]==0.34.2
gradio==5.23.3
sqlalchemy==2.0.40
chromadb==0.6.3
openai==1.75.0
python-multipart==0.0.20
python-dotenv==1.1.0
sentence-transformers==4.1.0
httpx==0.28.1
pytest==8.3.5
pytest-asyncio==0.25.3
```

- [ ] **Step 2: Write .env.example**

```
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=llama3.1:8b
LLM_API_KEY=ollama
TWIN_NAME=auto
CHROMADB_PATH=./data/chromadb
SQLITE_PATH=./db/chat_history.db
EMBEDDING_MODEL=all-MiniLM-L6-v2
DATA_DIR=./data
```

- [ ] **Step 3: Copy .env.example to .env**

```bash
cp .env.example .env
```

For local development, edit `.env` to set `LLM_BASE_URL=http://localhost:11434/v1`.

- [ ] **Step 4: Write .gitignore**

```
__pycache__/
*.pyc
.env
data/
db/
*.egg-info/
dist/
build/
.venv/
venv/
*.zip
inbox/
```

- [ ] **Step 5: Write app/__init__.py**

Empty file.

- [ ] **Step 6: Write tests/__init__.py**

Empty file.

- [ ] **Step 7: Write the failing test for config**

```python
# tests/test_config.py
import os
from app.config import Settings


def test_settings_defaults():
    """Settings loads with sensible defaults when no env vars set."""
    settings = Settings()
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.chromadb_path == "./data/chromadb"
    assert settings.sqlite_path == "./db/chat_history.db"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.data_dir == "./data"


def test_settings_from_env(monkeypatch):
    """Settings reads from environment variables."""
    monkeypatch.setenv("LLM_BASE_URL", "http://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")
    settings = Settings()
    assert settings.llm_base_url == "http://example.com/v1"
    assert settings.llm_model == "gpt-4"
```

- [ ] **Step 8: Run test to verify it fails**

Run: `cd /Users/hoangquocvietuet/Projects/digital-twins && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 9: Write app/config.py**

```python
"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "ollama")
    )
    twin_name: str = field(
        default_factory=lambda: os.getenv("TWIN_NAME", "auto")
    )
    chromadb_path: str = field(
        default_factory=lambda: os.getenv("CHROMADB_PATH", "./data/chromadb")
    )
    sqlite_path: str = field(
        default_factory=lambda: os.getenv("SQLITE_PATH", "./db/chat_history.db")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    data_dir: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "./data")
    )


settings = Settings()
```

- [ ] **Step 10: Write tests/conftest.py**

```python
"""Shared test fixtures."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with sample fingerprint and chunks.

    Structure: tmp_path/hoang_quoc_viet/ (this IS the data_dir level).
    """
    target_dir = tmp_path / "hoang_quoc_viet"
    target_dir.mkdir()

    # Sample style fingerprint
    fingerprint = {
        "total_messages": 940,
        "avg_length": 38.4,
        "median_length": 25,
        "avg_words_per_msg": 9.2,
        "length_distribution": {"6-20": 385, "21-50": 424},
        "punctuation": {
            "all_lowercase_pct": 71.1,
            "ends_with_period_pct": 3.4,
            "uses_exclamation_pct": 1.3,
            "question_mark_pct": 4.4,
            "uses_ellipsis_pct": 0.3,
            "has_emoji_pct": 0.9,
        },
        "top_emojis": [["🤣", 3]],
        "top_words": [["anh", 146], ["mình", 102], ["cho", 82]],
    }
    with open(target_dir / "style_fingerprint.json", "w") as f:
        json.dump(fingerprint, f)

    # Sample train chunks (3 chunks)
    chunks = [
        {
            "chunk_id": "dm_test_0",
            "thread_id": "inbox/test_thread",
            "chunk_type": "dm",
            "score": 1.5,
            "context": [
                {"author": "Friend", "text": "bạn làm gì đấy?", "timestamp": "2025-08-01T10:00:00", "is_target": False}
            ],
            "response": {"author": "Hoàng Quốc Việt", "text": "đang code dự án mới nè", "timestamp": "2025-08-01T10:01:00"},
            "response_length": 23,
            "context_turns": 1,
            "has_question": True,
            "time_gap_seconds": 0.0,
        },
        {
            "chunk_id": "dm_test_1",
            "thread_id": "inbox/test_thread",
            "chunk_type": "dm",
            "score": 1.2,
            "context": [
                {"author": "Friend", "text": "đi ăn không?", "timestamp": "2025-08-01T12:00:00", "is_target": False}
            ],
            "response": {"author": "Hoàng Quốc Việt", "text": "ok đi anh", "timestamp": "2025-08-01T12:01:00"},
            "response_length": 10,
            "context_turns": 1,
            "has_question": True,
            "time_gap_seconds": 0.0,
        },
        {
            "chunk_id": "dm_test_2",
            "thread_id": "inbox/test_thread_2",
            "chunk_type": "dm",
            "score": 0.8,
            "context": [],
            "response": {"author": "Hoàng Quốc Việt", "text": "ê bạn ơi", "timestamp": "2025-08-02T09:00:00"},
            "response_length": 9,
            "context_turns": 0,
            "has_question": False,
            "time_gap_seconds": 0.0,
        },
    ]
    with open(target_dir / "train_chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return tmp_path


@pytest.fixture
def sample_fingerprint(tmp_data_dir):
    """Return path to the sample fingerprint file."""
    return str(tmp_data_dir / "hoang_quoc_viet" / "style_fingerprint.json")


@pytest.fixture
def sample_chunks_path(tmp_data_dir):
    """Return path to the sample train chunks file."""
    return str(tmp_data_dir / "hoang_quoc_viet" / "train_chunks.jsonl")
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `cd /Users/hoangquocvietuet/Projects/digital-twins && pip install -r requirements.txt && python -m pytest tests/test_config.py -v`
Expected: 2 tests PASS

- [ ] **Step 12: Commit**

```bash
git init
git add requirements.txt .env.example .gitignore app/__init__.py app/config.py tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat: project scaffolding with config and test fixtures"
```

---

### Task 2: SQLite Database + ChatMessage Model

**Files:**
- Create: `app/database.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_database.py
import tempfile
import os
from app.database import create_engine_and_tables, ChatMessage, SessionFactory


def test_create_tables_and_insert(tmp_path):
    """Can create tables and insert a chat message."""
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    factory = SessionFactory(engine)

    with factory() as session:
        msg = ChatMessage(
            twin_slug="hoang_quoc_viet",
            role="user",
            content="xin chào",
        )
        session.add(msg)
        session.commit()

        result = session.query(ChatMessage).first()
        assert result.content == "xin chào"
        assert result.role == "user"
        assert result.twin_slug == "hoang_quoc_viet"
        assert result.id == 1
        assert result.created_at is not None


def test_retrieval_metadata_json(tmp_path):
    """retrieval_metadata stores JSON data."""
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    factory = SessionFactory(engine)

    with factory() as session:
        msg = ChatMessage(
            twin_slug="hoang_quoc_viet",
            role="assistant",
            content="đang code nè",
            retrieval_metadata={"chunks": 3, "avg_similarity": 0.85},
            tokens_used=42,
        )
        session.add(msg)
        session.commit()

        result = session.query(ChatMessage).first()
        assert result.retrieval_metadata["chunks"] == 3
        assert result.tokens_used == 42


def test_get_recent_messages(tmp_path):
    """Can query recent messages in chronological order."""
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    factory = SessionFactory(engine)

    with factory() as session:
        for i in range(15):
            session.add(ChatMessage(
                twin_slug="hoang_quoc_viet",
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
            ))
        session.commit()

        recent = (
            session.query(ChatMessage)
            .filter_by(twin_slug="hoang_quoc_viet")
            .order_by(ChatMessage.id.desc())
            .limit(10)
            .all()
        )
        recent.reverse()
        assert len(recent) == 10
        assert recent[0].content == "message 5"
        assert recent[-1].content == "message 14"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.database'`

- [ ] **Step 3: Write app/database.py**

```python
"""SQLite database setup for chat history.

Uses a session factory pattern for thread-safety with uvicorn's thread pool.
Each request/operation creates its own session via the factory.
"""

import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    twin_slug = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    retrieval_metadata = Column(JSON, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_engine_and_tables(db_path: str):
    """Create SQLite engine and ensure tables exist."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


class SessionFactory:
    """Thread-safe session factory. Creates a new session per call.

    Usage:
        factory = SessionFactory(engine)
        with factory() as session:
            session.query(...)
    """

    def __init__(self, engine):
        self._sessionmaker = sessionmaker(bind=engine)

    @contextmanager
    def __call__(self):
        session = self._sessionmaker()
        try:
            yield session
        finally:
            session.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_database.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/database.py tests/test_database.py
git commit -m "feat: SQLite chat history with thread-safe session factory"
```

---

### Task 3: System Prompt Builder

**Files:**
- Create: `app/prompt.py`
- Create: `tests/test_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt.py
from app.prompt import build_system_prompt, load_fingerprint


def test_load_fingerprint(sample_fingerprint):
    """Loads fingerprint JSON from disk."""
    fp = load_fingerprint(sample_fingerprint)
    assert fp["total_messages"] == 940
    assert fp["avg_length"] == 38.4
    assert fp["punctuation"]["all_lowercase_pct"] == 71.1


def test_build_system_prompt_contains_key_rules(sample_fingerprint):
    """System prompt includes fingerprint-derived rules."""
    fp = load_fingerprint(sample_fingerprint)
    prompt = build_system_prompt("Hoàng Quốc Việt", fp)

    assert "Hoàng Quốc Việt" in prompt
    assert "71.1%" in prompt  # all_lowercase_pct
    assert "38.4" in prompt  # avg_length
    assert "25" in prompt  # median_length
    assert "3.4%" in prompt  # ends_with_period_pct
    assert "0.9%" in prompt  # has_emoji_pct
    assert "9.2" in prompt  # avg_words_per_msg
    assert "anh" in prompt  # top word
    assert "NOT an AI assistant" in prompt


def test_build_system_prompt_no_fingerprint():
    """Falls back to a generic prompt when no fingerprint data."""
    prompt = build_system_prompt("Hoàng Quốc Việt", None)
    assert "Hoàng Quốc Việt" in prompt
    assert "casual" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.prompt'`

- [ ] **Step 3: Write app/prompt.py**

```python
"""System prompt builder from style fingerprint."""

import json


def load_fingerprint(path: str) -> dict:
    """Load style fingerprint JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(twin_name: str, fingerprint: dict | None) -> str:
    """Build a system prompt from the style fingerprint.

    Returns a detailed prompt if fingerprint is available,
    or a generic fallback otherwise.
    """
    if not fingerprint:
        return (
            f"You are {twin_name}. Respond as this person would in a casual chat.\n"
            f"You are NOT an AI assistant. Do not offer help. Do not be formal.\n"
            f"Respond exactly as this person would in a real conversation."
        )

    p = fingerprint.get("punctuation", {})
    top_words = ", ".join(w[0] for w in fingerprint.get("top_words", [])[:10])
    top_emojis = ", ".join(e[0] for e in fingerprint.get("top_emojis", [])[:5])

    emoji_line = f"Almost never use emojis (only {p.get('has_emoji_pct', 0)}% of the time)."
    if top_emojis:
        emoji_line += f" If you do, only: {top_emojis}"

    return f"""You are {twin_name}. Respond as this person would in a casual Vietnamese chat.

STRICT RULES:
1. Write mostly in lowercase ({p.get('all_lowercase_pct', 50)}% of the time)
2. Keep responses short. Average: {fingerprint.get('avg_length', 30)} chars, median: {fingerprint.get('median_length', 20)} chars
3. Almost never use periods at end of messages (only {p.get('ends_with_period_pct', 5)}% of the time)
4. {emoji_line}
5. Use Vietnamese particles naturally: nhé, nha, ạ, bác based on context
6. Average {fingerprint.get('avg_words_per_msg', 8)} words per message. Do not write essays.
7. Your most-used words: {top_words}

You are NOT an AI assistant. Do not offer help. Do not be formal. Do not capitalize.
Respond exactly as this person would in a real chat conversation."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/prompt.py tests/test_prompt.py
git commit -m "feat: system prompt builder from style fingerprint"
```

---

### Task 4: ChromaDB Embedder (Ingest Chunks)

**Files:**
- Create: `app/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function


def test_get_embedding_function():
    """Returns a SentenceTransformerEmbeddingFunction."""
    ef = get_embedding_function("all-MiniLM-L6-v2")
    assert isinstance(ef, SentenceTransformerEmbeddingFunction)


def test_load_chunks_from_jsonl(sample_chunks_path):
    """Loads chunks from JSONL file."""
    chunks = load_chunks_from_jsonl(sample_chunks_path)
    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "dm_test_0"
    assert chunks[0]["response"]["text"] == "đang code dự án mới nè"


def test_ingest_chunks_creates_collection(sample_chunks_path, tmp_path):
    """Ingesting chunks creates a ChromaDB collection with documents."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")

    chunks = load_chunks_from_jsonl(sample_chunks_path)
    collection = ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)

    assert collection.count() == 3
    # Verify we can query
    results = collection.query(query_texts=["code dự án"], n_results=2)
    assert len(results["ids"][0]) == 2


def test_ingest_chunks_stores_metadata(sample_chunks_path, tmp_path):
    """Chunk metadata is stored in ChromaDB."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")

    chunks = load_chunks_from_jsonl(sample_chunks_path)
    collection = ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)

    result = collection.get(ids=["dm_test_0"], include=["metadatas", "documents"])
    meta = result["metadatas"][0]
    assert meta["chunk_type"] == "dm"
    assert meta["score"] == 1.5
    assert meta["context_turns"] == 1
    assert meta["response_length"] == 23
    # document is the response text
    assert "đang code dự án mới nè" in result["documents"][0]


def test_ingest_overwrites_existing_collection(sample_chunks_path, tmp_path):
    """Re-ingesting deletes the old collection and creates a new one."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")

    chunks = load_chunks_from_jsonl(sample_chunks_path)
    ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)
    assert client.get_collection("hoang_quoc_viet", embedding_function=ef).count() == 3

    # Re-ingest with only 2 chunks
    ingest_chunks(client, "hoang_quoc_viet", chunks[:2], embedding_function=ef)
    assert client.get_collection("hoang_quoc_viet", embedding_function=ef).count() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.embedder'`

- [ ] **Step 3: Write app/embedder.py**

```python
"""ChromaDB ingestion from train_chunks.jsonl.

Uses sentence-transformers for embeddings. The embedding model is configured
via EMBEDDING_MODEL in .env. Changing it requires re-embedding all chunks.
"""

import json

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def get_embedding_function(model_name: str) -> SentenceTransformerEmbeddingFunction:
    """Create a sentence-transformer embedding function for ChromaDB."""
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def load_chunks_from_jsonl(path: str) -> list[dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _chunk_to_document(chunk: dict) -> str:
    """Build the document text that gets embedded.

    Includes context + response so retrieval matches both
    the question pattern and the response style.
    """
    parts = []
    for ctx in chunk.get("context", []):
        parts.append(f"{ctx['author']}: {ctx['text']}")
    parts.append(f"{chunk['response']['author']}: {chunk['response']['text']}")
    return "\n".join(parts)


def ingest_chunks(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: list[dict],
    embedding_function: SentenceTransformerEmbeddingFunction | None = None,
) -> chromadb.Collection:
    """Ingest chunks into a ChromaDB collection.

    Deletes existing collection if present (overwrite mode).
    embedding_function must be provided to ensure consistent embeddings.
    """
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )

    # Batch add
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
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

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/embedder.py tests/test_embedder.py
git commit -m "feat: ChromaDB embedder with explicit sentence-transformer model"
```

---

### Task 5: ChromaDB Retrieval

**Files:**
- Create: `app/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieval.py
import chromadb
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function
from app.retrieval import retrieve_chunks, format_few_shot_examples


def test_retrieve_chunks_returns_results(sample_chunks_path, tmp_path):
    """Retrieval returns ranked chunks for a query."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")
    chunks = load_chunks_from_jsonl(sample_chunks_path)
    collection = ingest_chunks(client, "test_twin", chunks, embedding_function=ef)

    results = retrieve_chunks(collection, "bạn đang làm gì", n_results=2)
    assert len(results) == 2
    assert "chunk_id" in results[0]
    assert "document" in results[0]
    assert "distance" in results[0]
    assert "metadata" in results[0]


def test_retrieve_chunks_empty_collection(tmp_path):
    """Retrieval on empty collection returns empty list."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection("empty_twin")

    results = retrieve_chunks(collection, "hello", n_results=5)
    assert results == []


def test_format_few_shot_examples():
    """Formats retrieved chunks as few-shot examples for the prompt."""
    retrieved = [
        {
            "chunk_id": "dm_test_0",
            "document": "Friend: bạn làm gì đấy?\nViệt: đang code dự án mới nè",
            "distance": 0.3,
            "metadata": {"chunk_type": "dm", "score": 1.5},
        },
        {
            "chunk_id": "dm_test_1",
            "document": "Friend: đi ăn không?\nViệt: ok đi anh",
            "distance": 0.5,
            "metadata": {"chunk_type": "dm", "score": 1.2},
        },
    ]
    examples = format_few_shot_examples(retrieved)
    assert "bạn làm gì đấy?" in examples
    assert "đang code dự án mới nè" in examples
    assert "đi ăn không?" in examples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval'`

- [ ] **Step 3: Write app/retrieval.py**

```python
"""ChromaDB query and context assembly for RAG."""

import chromadb


def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """Query ChromaDB collection and return ranked results.

    Returns list of dicts with: chunk_id, document, distance, metadata.
    Returns empty list if collection is empty.
    """
    if collection.count() == 0:
        return []

    # Don't request more results than documents exist
    actual_n = min(n_results, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=actual_n,
        include=["documents", "distances", "metadatas"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return chunks


def format_few_shot_examples(retrieved: list[dict], max_examples: int = 3) -> str:
    """Format retrieved chunks as few-shot examples for the LLM prompt.

    Each example shows a conversation snippet the twin actually had.
    """
    if not retrieved:
        return ""

    examples = []
    for chunk in retrieved[:max_examples]:
        examples.append(chunk["document"])

    return "\n\n---\n\n".join(examples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "feat: ChromaDB retrieval with few-shot formatting"
```

---

### Task 6: Chat Service (Shared Logic)

**Files:**
- Create: `app/chat_service.py`
- Create: `tests/test_chat_service.py`

This is the shared chat logic used by both the API endpoint and the Gradio UI. No duplication.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_service.py
import json
from unittest.mock import MagicMock, patch

import pytest

from app.chat_service import chat, ChatResult


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection that returns test results."""
    collection = MagicMock()
    collection.count.return_value = 3
    collection.query.return_value = {
        "ids": [["dm_test_0"]],
        "documents": [["Friend: bạn làm gì?\nViệt: đang code nè"]],
        "distances": [[0.3]],
        "metadatas": [[{"chunk_type": "dm", "score": 1.5}]],
    }
    return collection


@pytest.fixture
def mock_session_factory():
    """Mock session factory that returns a mock session."""
    session = MagicMock()
    session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

    factory = MagicMock()
    factory.return_value.__enter__ = MagicMock(return_value=session)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    return factory


def test_chat_rejects_empty_message(mock_collection, mock_session_factory):
    """Empty content returns error result."""
    result = chat(
        content="",
        collection=mock_collection,
        session_factory=mock_session_factory,
        twin_slug="test",
        system_prompt="You are Việt.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    assert result.error is True
    assert "message" in result.content.lower()


def test_chat_returns_no_data_when_empty_collection(mock_session_factory):
    """When collection is empty, returns import prompt."""
    empty_collection = MagicMock()
    empty_collection.count.return_value = 0

    result = chat(
        content="hello",
        collection=empty_collection,
        session_factory=mock_session_factory,
        twin_slug="test",
        system_prompt="You are Việt.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    assert "import data" in result.content.lower() or "not enough context" in result.content.lower()


def test_chat_truncates_and_calls_llm(mock_collection, mock_session_factory):
    """Long messages are truncated; LLM is called with truncated content."""
    long_msg = "a" * 15000

    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "đang code nè"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        result = chat(
            content=long_msg,
            collection=mock_collection,
            session_factory=mock_session_factory,
            twin_slug="test",
            system_prompt="You are Việt.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
            llm_api_key="ollama",
        )

        assert result.content == "đang code nè"
        assert result.error is False

        # Verify the user message in the LLM call was truncated to 10K
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][-1]
        assert len(user_msg["content"]) == 10_000


def test_chat_handles_connection_error(mock_collection, mock_session_factory):
    """Connection error returns friendly message."""
    import openai as openai_lib
    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai_lib.APIConnectionError(
            request=MagicMock()
        )

        result = chat(
            content="hello",
            collection=mock_collection,
            session_factory=mock_session_factory,
            twin_slug="test",
            system_prompt="You are Việt.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
            llm_api_key="ollama",
        )
        assert "ollama" in result.content.lower() or "reach" in result.content.lower()
        assert result.error is True


def test_chat_handles_json_decode_error(mock_collection, mock_session_factory):
    """Malformed LLM JSON response returns friendly message."""
    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = json.JSONDecodeError(
            "Expecting value", "", 0
        )

        result = chat(
            content="hello",
            collection=mock_collection,
            session_factory=mock_session_factory,
            twin_slug="test",
            system_prompt="You are Việt.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
            llm_api_key="ollama",
        )
        assert "unexpected" in result.content.lower() or "format" in result.content.lower()
        assert result.error is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chat_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chat_service'`

- [ ] **Step 3: Write app/chat_service.py**

```python
"""Shared chat logic used by both the API endpoint and the Gradio UI.

Single source of truth for: validation, retrieval, prompt building,
LLM call, error handling, and DB persistence.
"""

import json
from dataclasses import dataclass

import openai

from app.database import ChatMessage
from app.retrieval import retrieve_chunks, format_few_shot_examples

MAX_MESSAGE_LENGTH = 10_000


@dataclass
class ChatResult:
    content: str
    retrieval_metadata: dict | None = None
    error: bool = False


def chat(
    content: str,
    collection,
    session_factory,
    twin_slug: str,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
) -> ChatResult:
    """Process a chat message through the full RAG pipeline.

    1. Validate input
    2. Retrieve similar chunks from ChromaDB
    3. Build prompt with system prompt + few-shot + history
    4. Call LLM
    5. Save to DB
    6. Return result
    """
    # 1. Validate
    content = content.strip()
    if not content:
        return ChatResult(content="Please enter a message.", error=True)

    # Truncate silently if too long
    if len(content) > MAX_MESSAGE_LENGTH:
        content = content[:MAX_MESSAGE_LENGTH]

    # 2. Check if twin has data
    if collection.count() == 0:
        return ChatResult(
            content="I don't have enough context to answer that authentically. Import data first.",
            error=True,
        )

    # 3. Retrieve
    retrieved = retrieve_chunks(collection, content, n_results=5)

    if not retrieved:
        return ChatResult(
            content="I don't have enough context to answer that authentically.",
            retrieval_metadata={"chunks": 0, "avg_similarity": 0},
            error=True,
        )

    # 4. Build messages
    few_shot = format_few_shot_examples(retrieved, max_examples=3)
    avg_distance = sum(r["distance"] for r in retrieved) / len(retrieved)

    # Get recent chat history
    with session_factory() as session:
        recent_msgs = (
            session.query(ChatMessage)
            .filter_by(twin_slug=twin_slug)
            .order_by(ChatMessage.id.desc())
            .limit(10)
            .all()
        )
        recent_msgs.reverse()

    messages = [{"role": "system", "content": system_prompt}]

    if few_shot:
        messages.append({
            "role": "system",
            "content": f"Here are examples of how you actually talk:\n\n{few_shot}",
        })

    for msg in recent_msgs:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": content})

    # 5. LLM call
    try:
        client = openai.OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            stream=False,
            timeout=30,
        )
        assistant_content = response.choices[0].message.content or ""
    except openai.APIConnectionError:
        return ChatResult(
            content="Could not reach the LLM. Check that Ollama is running.",
            error=True,
        )
    except openai.APITimeoutError:
        return ChatResult(
            content="LLM took too long to respond. Try again.",
            error=True,
        )
    except openai.RateLimitError:
        return ChatResult(
            content="Rate limited. Wait a moment and try again.",
            error=True,
        )
    except json.JSONDecodeError:
        return ChatResult(
            content="Unexpected response format from LLM.",
            error=True,
        )

    if not assistant_content.strip():
        return ChatResult(
            content="No response generated. Try rephrasing.",
            error=True,
        )

    retrieval_meta = {
        "chunks": len(retrieved),
        "avg_similarity": round(1 - avg_distance, 3),
    }

    # 6. Save to DB
    try:
        with session_factory() as session:
            session.add(ChatMessage(
                twin_slug=twin_slug,
                role="user",
                content=content,
            ))
            session.add(ChatMessage(
                twin_slug=twin_slug,
                role="assistant",
                content=assistant_content,
                retrieval_metadata=retrieval_meta,
                tokens_used=response.usage.total_tokens if response.usage else None,
            ))
            session.commit()
    except Exception:
        pass  # Response still shown, history not saved

    return ChatResult(
        content=assistant_content,
        retrieval_metadata=retrieval_meta,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chat_service.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/chat_service.py tests/test_chat_service.py
git commit -m "feat: shared chat service with RAG, error handling, and DB persistence"
```

---

### Task 7: Chat API Endpoint + Export

**Files:**
- Create: `app/chat.py`
- Create: `tests/test_chat.py`

Thin wrapper around `chat_service.py`. Also includes `GET /api/export`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat.py
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat import create_chat_router


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection that returns test results."""
    collection = MagicMock()
    collection.count.return_value = 3
    collection.query.return_value = {
        "ids": [["dm_test_0"]],
        "documents": [["Friend: bạn làm gì?\nViệt: đang code nè"]],
        "distances": [[0.3]],
        "metadatas": [[{"chunk_type": "dm", "score": 1.5}]],
    }
    return collection


@pytest.fixture
def mock_session_factory():
    """Mock session factory."""
    session = MagicMock()
    session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
    session.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []

    factory = MagicMock()
    factory.return_value.__enter__ = MagicMock(return_value=session)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    return factory


@pytest.fixture
def chat_app(mock_collection, mock_session_factory):
    """FastAPI app with chat router for testing."""
    app = FastAPI()
    router = create_chat_router(
        collection=mock_collection,
        session_factory=mock_session_factory,
        twin_slug="hoang_quoc_viet",
        system_prompt="You are Việt. Respond casually.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    app.include_router(router)
    return app


def test_chat_rejects_empty_message(chat_app):
    """POST /api/chat with empty content returns 400."""
    client = TestClient(chat_app)
    resp = client.post("/api/chat", json={"content": ""})
    assert resp.status_code == 400
    assert "message" in resp.json()["detail"].lower()


def test_chat_returns_no_twin_error_when_empty_collection(mock_session_factory):
    """When ChromaDB collection is empty, returns appropriate error."""
    empty_collection = MagicMock()
    empty_collection.count.return_value = 0

    app = FastAPI()
    router = create_chat_router(
        collection=empty_collection,
        session_factory=mock_session_factory,
        twin_slug="hoang_quoc_viet",
        system_prompt="You are Việt.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/api/chat", json={"content": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert "not enough context" in body["content"].lower() or "import data" in body["content"].lower()


def test_chat_success_with_mocked_llm(chat_app, mock_collection):
    """Successful chat returns LLM response with retrieval metadata."""
    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "đang code nè bạn"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        client = TestClient(chat_app)
        resp = client.post("/api/chat", json={"content": "bạn đang làm gì?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "đang code nè bạn"
        assert body["retrieval_metadata"]["chunks"] > 0


def test_export_empty_history(chat_app):
    """GET /api/export with no messages returns empty list."""
    client = TestClient(chat_app)
    resp = client.get("/api/export")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chat'`

- [ ] **Step 3: Write app/chat.py**

```python
"""Chat and export API endpoints. Thin wrappers around chat_service."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.chat_service import chat as chat_service_fn, ChatResult
from app.database import ChatMessage


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    content: str
    retrieval_metadata: dict | None = None


def create_chat_router(
    collection,
    session_factory,
    twin_slug: str,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        content = req.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="Please enter a message.")

        result = chat_service_fn(
            content=req.content,
            collection=collection,
            session_factory=session_factory,
            twin_slug=twin_slug,
            system_prompt=system_prompt,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
        return ChatResponse(
            content=result.content,
            retrieval_metadata=result.retrieval_metadata,
        )

    @router.get("/api/export")
    def export_chat():
        """Export chat history as JSON."""
        with session_factory() as session:
            messages = (
                session.query(ChatMessage)
                .filter_by(twin_slug=twin_slug)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "retrieval_metadata": msg.retrieval_metadata,
                }
                for msg in messages
            ]

    return router
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chat.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/chat.py tests/test_chat.py
git commit -m "feat: chat and export API endpoints"
```

---

### Task 8: Import Endpoint

**Files:**
- Create: `app/importer.py`
- Create: `tests/test_importer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_importer.py
import json
import os
import tempfile
import zipfile

import pytest

from app.importer import (
    validate_zip,
    find_inbox_folder,
    ZipValidationError,
)


def _create_test_zip(tmp_path, with_inbox=True) -> str:
    """Create a minimal test zip with Facebook-like structure."""
    zip_path = str(tmp_path / "test_export.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        if with_inbox:
            # Create a minimal conversation
            conv_data = {
                "participants": [
                    {"name": "Ho\u00c3\u00a0ng Qu\u00e1\u00bb\u0091c Vi\u00e1\u00bb\u0087t"},
                    {"name": "Friend"},
                ],
                "messages": [
                    {
                        "sender_name": "Friend",
                        "timestamp_ms": 1690000000000,
                        "content": "xin ch\u00c3\u00a0o",
                    },
                    {
                        "sender_name": "Ho\u00c3\u00a0ng Qu\u00e1\u00bb\u0091c Vi\u00e1\u00bb\u0087t",
                        "timestamp_ms": 1690000060000,
                        "content": "ch\u00c3\u00a0o b\u00e1\u00ba\u00a1n",
                    },
                    {
                        "sender_name": "Friend",
                        "timestamp_ms": 1690000120000,
                        "content": "b\u00e1\u00ba\u00a1n kho\u00e1\u00bb\u008fe kh\u00c3\u00b4ng?",
                    },
                    {
                        "sender_name": "Ho\u00c3\u00a0ng Qu\u00e1\u00bb\u0091c Vi\u00e1\u00bb\u0087t",
                        "timestamp_ms": 1690000180000,
                        "content": "m\u00c3\u00acnh kho\u00e1\u00bb\u008fe nh\u00c3\u00a9",
                    },
                ],
            }
            zf.writestr(
                "inbox/friend_123/message_1.json",
                json.dumps(conv_data, ensure_ascii=False),
            )
    return zip_path


def test_validate_zip_accepts_valid(tmp_path):
    """Valid zip file passes validation."""
    zip_path = _create_test_zip(tmp_path)
    assert validate_zip(zip_path) is True


def test_validate_zip_rejects_non_zip(tmp_path):
    """Non-zip file raises ZipValidationError."""
    bad_path = str(tmp_path / "not_a_zip.txt")
    with open(bad_path, "w") as f:
        f.write("not a zip")
    with pytest.raises(ZipValidationError, match="zip"):
        validate_zip(bad_path)


def test_validate_zip_rejects_oversized(tmp_path):
    """Zip over 500MB raises ZipValidationError."""
    zip_path = _create_test_zip(tmp_path)
    with pytest.raises(ZipValidationError, match="large"):
        validate_zip(zip_path, max_size_mb=0)  # 0 MB limit = always too large


def test_find_inbox_folder_direct(tmp_path):
    """Finds inbox/ folder at top level."""
    inbox_dir = tmp_path / "inbox" / "friend_123"
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "message_1.json").write_text("{}")

    result = find_inbox_folder(str(tmp_path))
    assert result.endswith("inbox")


def test_find_inbox_folder_nested(tmp_path):
    """Finds inbox/ folder nested one or two levels deep."""
    inbox_dir = tmp_path / "your_facebook_activity" / "messages" / "inbox" / "friend_123"
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "message_1.json").write_text("{}")

    result = find_inbox_folder(str(tmp_path))
    assert result.endswith("inbox")


def test_find_inbox_folder_missing(tmp_path):
    """Returns None when no inbox/ folder exists."""
    result = find_inbox_folder(str(tmp_path))
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_importer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.importer'`

- [ ] **Step 3: Write app/importer.py**

```python
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

        # Run audit — wrap in try/except to catch sys.exit() from audit_facebook
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

        # Save audit outputs directly to data_dir (not nested)
        # audit_facebook.save_outputs adds "data/<safe_name>" to base_dir,
        # so we pass the PARENT of data_dir as base_dir.
        base_dir_for_audit = os.path.dirname(os.path.abspath(data_dir))
        # But if data_dir is already absolute and named "data", this works.
        # Safer: just write directly ourselves.
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

        # Check if collection already exists (warn user)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_importer.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/importer.py tests/test_importer.py
git commit -m "feat: import pipeline with consistent data_dir paths and sys.exit protection"
```

---

### Task 9: Gradio UI (Chat + Import Tabs)

**Files:**
- Create: `app/ui.py`
- Create: `tests/test_ui.py`

Uses `chat_service.py` — no duplicated chat logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ui.py
from unittest.mock import MagicMock, patch
from app.ui import create_ui


def test_create_ui_returns_gradio_blocks():
    """create_ui returns a Gradio Blocks instance."""
    import gradio as gr

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_session_factory = MagicMock()

    ui = create_ui(
        collection=mock_collection,
        session_factory=mock_session_factory,
        twin_slug="hoang_quoc_viet",
        system_prompt="You are Việt.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
        chromadb_client=MagicMock(),
        data_dir="./data",
        embedding_model="all-MiniLM-L6-v2",
    )
    assert isinstance(ui, gr.Blocks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ui'`

- [ ] **Step 3: Write app/ui.py**

```python
"""Gradio UI with Chat and Import tabs.

Uses chat_service for chat logic — no duplication with chat.py.
"""

import json
import os
import tempfile

import gradio as gr

from app.chat_service import chat as chat_service_fn
from app.database import ChatMessage
from app.importer import run_import_pipeline, ZipValidationError


def create_ui(
    collection,
    session_factory,
    twin_slug: str,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    chromadb_client,
    data_dir: str,
    embedding_model: str,
) -> gr.Blocks:
    """Create the Gradio UI with Chat and Import tabs."""

    def chat_fn(message: str, history: list[dict]) -> str:
        """Handle chat messages via shared chat_service."""
        result = chat_service_fn(
            content=message,
            collection=collection,
            session_factory=session_factory,
            twin_slug=twin_slug,
            system_prompt=system_prompt,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )

        if result.error:
            return result.content

        quality = f"\n\n_Matched {result.retrieval_metadata['chunks']} chunks (avg similarity: {result.retrieval_metadata['avg_similarity']})_"
        return result.content + quality

    def import_fn(file, target_name, progress=gr.Progress()):
        """Handle file import."""
        if file is None:
            return "Please upload a .zip file."

        def on_progress(msg):
            progress(0, desc=msg)

        try:
            result = run_import_pipeline(
                zip_path=file.name if hasattr(file, "name") else file,
                chromadb_client=chromadb_client,
                data_dir=data_dir,
                embedding_model=embedding_model,
                target_name=target_name if target_name else None,
                on_progress=on_progress,
            )
            overwrite_note = " (previous twin data was overwritten)" if result.get("overwritten") else ""
            return (
                f"Twin ready! **{result['twin_name']}**{overwrite_note}\n\n"
                f"- Messages: {result['total_messages']}\n"
                f"- Chunks embedded: {result['chunks_embedded']}\n\n"
                f"Switch to the Chat tab to start talking!"
            )
        except ZipValidationError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Failed to build twin. Error: {e}"

    def export_fn():
        """Export chat history as JSON."""
        with session_factory() as session:
            messages = (
                session.query(ChatMessage)
                .filter_by(twin_slug=twin_slug)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            if not messages:
                return None

            export = []
            for msg in messages:
                export.append({
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "retrieval_metadata": msg.retrieval_metadata,
                })

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="chat_export_"
        )
        json.dump(export, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        return tmp.name

    # Build UI
    with gr.Blocks(title="Digital Twins", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Digital Twins")

        with gr.Tab("Chat"):
            chatbot = gr.ChatInterface(
                fn=chat_fn,
                type="messages",
                title=None,
                description="Chat with your digital twin",
            )
            export_btn = gr.Button("Export Conversation", size="sm")
            export_file = gr.File(label="Download", visible=False)
            export_btn.click(fn=export_fn, outputs=export_file).then(
                fn=lambda f: gr.update(visible=f is not None),
                inputs=export_file,
                outputs=export_file,
            )

        with gr.Tab("Import"):
            gr.Markdown("### Import Facebook Messenger Data")
            gr.Markdown("Upload your Facebook data export (.zip) to build a twin.")
            upload = gr.File(label="Upload .zip file", file_types=[".zip"])
            target_input = gr.Textbox(
                label="Target name (optional)",
                placeholder="Leave empty to auto-detect from data",
            )
            import_btn = gr.Button("Build Twin", variant="primary")
            import_output = gr.Markdown(label="Status")
            import_btn.click(
                fn=import_fn,
                inputs=[upload, target_input],
                outputs=import_output,
            )

    return demo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui.py -v`
Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
git add app/ui.py tests/test_ui.py
git commit -m "feat: Gradio UI using shared chat_service, no duplication"
```

---

### Task 10: FastAPI App + Main Entrypoint

**Files:**
- Create: `app/main.py`

- [ ] **Step 1: Write app/main.py**

```python
"""FastAPI app with Gradio UI mount."""

import os

import chromadb
import gradio as gr
from fastapi import FastAPI

from app.config import settings
from app.database import create_engine_and_tables, SessionFactory
from app.chat import create_chat_router
from app.embedder import get_embedding_function
from app.prompt import build_system_prompt, load_fingerprint
from app.ui import create_ui


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Digital Twins")

    # Database — session factory for thread-safe access
    engine = create_engine_and_tables(settings.sqlite_path)
    session_factory = SessionFactory(engine)

    # ChromaDB with explicit embedding model
    os.makedirs(settings.chromadb_path, exist_ok=True)
    chromadb_client = chromadb.PersistentClient(path=settings.chromadb_path)
    embedding_fn = get_embedding_function(settings.embedding_model)

    # Detect twin — scan data_dir for directories with a fingerprint
    twin_slug = None
    system_prompt = "You are a digital twin. Import data to get started."
    collection = None

    if os.path.isdir(settings.data_dir):
        for name in os.listdir(settings.data_dir):
            fp_path = os.path.join(settings.data_dir, name, "style_fingerprint.json")
            if os.path.isfile(fp_path):
                twin_slug = name
                break

    if twin_slug:
        fp_path = os.path.join(settings.data_dir, twin_slug, "style_fingerprint.json")
        fingerprint = load_fingerprint(fp_path)

        # Determine display name
        twin_name = settings.twin_name
        if twin_name == "auto":
            twin_name = twin_slug.replace("_", " ").title()

        system_prompt = build_system_prompt(twin_name, fingerprint)

        # Get or create ChromaDB collection with explicit embedding function
        try:
            collection = chromadb_client.get_collection(twin_slug, embedding_function=embedding_fn)
        except ValueError:
            collection = chromadb_client.get_or_create_collection(
                twin_slug, embedding_function=embedding_fn
            )

        # If collection is empty but chunks exist, auto-embed
        if collection.count() == 0:
            chunks_path = os.path.join(settings.data_dir, twin_slug, "train_chunks.jsonl")
            if os.path.isfile(chunks_path):
                from app.embedder import load_chunks_from_jsonl, ingest_chunks
                chunks = load_chunks_from_jsonl(chunks_path)
                collection = ingest_chunks(
                    chromadb_client, twin_slug, chunks, embedding_function=embedding_fn
                )
    else:
        twin_slug = "default"
        collection = chromadb_client.get_or_create_collection(
            "default", embedding_function=embedding_fn
        )

    # Chat API + Export API
    chat_router = create_chat_router(
        collection=collection,
        session_factory=session_factory,
        twin_slug=twin_slug,
        system_prompt=system_prompt,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
    )
    app.include_router(chat_router)

    # Gradio UI
    ui = create_ui(
        collection=collection,
        session_factory=session_factory,
        twin_slug=twin_slug,
        system_prompt=system_prompt,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
        chromadb_client=chromadb_client,
        data_dir=settings.data_dir,
        embedding_model=settings.embedding_model,
    )
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
```

- [ ] **Step 2: Verify the app starts without errors**

Run: `cd /Users/hoangquocvietuet/Projects/digital-twins && timeout 5 python -c "from app.main import create_app; print('App created successfully')" || true`
Expected: "App created successfully" (may fail on missing data dir, that's ok — verify no import errors)

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: FastAPI main entrypoint with explicit embedding model and session factory"
```

---

### Task 11: Docker Deployment

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write Dockerfile**

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

- [ ] **Step 2: Write docker-compose.yml**

```yaml
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

- [ ] **Step 3: Verify Docker build (if Docker available)**

Run: `cd /Users/hoangquocvietuet/Projects/digital-twins && docker build -t digital-twins . 2>&1 | tail -5`
Expected: "Successfully built" or "Successfully tagged"

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Docker single-container deployment"
```

---

### Task 12: Integration Test (Full Chat Pipeline)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: full pipeline from chunks to chat response."""

import json
import os
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat import create_chat_router
from app.database import create_engine_and_tables, SessionFactory, ChatMessage
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function
from app.prompt import build_system_prompt, load_fingerprint
from app.retrieval import retrieve_chunks


def test_full_chat_pipeline(tmp_data_dir, tmp_path):
    """End-to-end: ingest chunks -> build prompt -> query -> get response."""
    # 1. Setup DB with session factory
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    session_factory = SessionFactory(engine)

    # 2. Load fingerprint and build system prompt
    fp_path = str(tmp_data_dir / "hoang_quoc_viet" / "style_fingerprint.json")
    fp = load_fingerprint(fp_path)
    system_prompt = build_system_prompt("Hoàng Quốc Việt", fp)
    assert "Hoàng Quốc Việt" in system_prompt

    # 3. Ingest chunks into ChromaDB with explicit embedding function
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")
    chunks_path = str(tmp_data_dir / "hoang_quoc_viet" / "train_chunks.jsonl")
    chunks = load_chunks_from_jsonl(chunks_path)
    collection = ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)
    assert collection.count() == 3

    # 4. Verify retrieval works
    retrieved = retrieve_chunks(collection, "đang làm gì", n_results=2)
    assert len(retrieved) > 0

    # 5. Test chat endpoint with mocked LLM
    app = FastAPI()
    router = create_chat_router(
        collection=collection,
        session_factory=session_factory,
        twin_slug="hoang_quoc_viet",
        system_prompt=system_prompt,
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
    )
    app.include_router(router)

    with patch("app.chat_service.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "đang code nè bạn"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        test_client = TestClient(app)
        resp = test_client.post("/api/chat", json={"content": "bạn đang làm gì thế?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "đang code nè bạn"
        assert body["retrieval_metadata"]["chunks"] > 0

    # 6. Verify chat was saved to DB
    with session_factory() as session:
        messages = session.query(ChatMessage).all()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].twin_slug == "hoang_quoc_viet"
        assert messages[1].role == "assistant"
        assert messages[1].content == "đang code nè bạn"

    # 7. Test export endpoint
    test_client2 = TestClient(app)
    resp = test_client2.get("/api/export")
    assert resp.status_code == 200
    exported = resp.json()
    assert len(exported) == 2
    assert exported[0]["role"] == "user"
    assert exported[1]["role"] == "assistant"
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: 1 test PASS

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full chat pipeline + export"
```

---

## Self-Review Checklist

### Spec coverage:
- Upload zip -> chunks in ChromaDB: **Task 4 (embedder) + Task 8 (importer)**
- prompt.py builds correct system prompt: **Task 3**
- retrieval.py returns ranked chunks: **Task 5**
- /api/chat returns response: **Task 7**
- GET /api/export: **Task 7**
- Gradio Chat tab (quality indicator + export): **Task 9**
- Gradio Import tab (upload -> audit -> build): **Task 9**
- SQLite chat history: **Task 2**
- Docker deployment: **Task 11**
- Error handling (LLM timeout, connection, rate limit, JSONDecode, zip slip): **Task 6 + Task 8**
- Config from .env: **Task 1**
- Embedding model explicitly configured: **Task 4, passed through all layers**
- Integration test: **Task 12**

### Placeholder scan: None found.

### Type consistency:
- `load_fingerprint` returns `dict` — used consistently in prompt.py and main.py
- `load_chunks_from_jsonl` returns `list[dict]` — used in embedder.py and importer.py
- `retrieve_chunks` returns `list[dict]` with keys: chunk_id, document, distance, metadata
- `ChatMessage` model used consistently with twin_slug, role, content fields
- `ChatResult` dataclass returned by `chat_service.chat()`, consumed by both `chat.py` and `ui.py`
- `SessionFactory` used everywhere instead of raw sessions — thread-safe
- `twin_slug` parameter flows from main.py through chat_router and ui consistently
- `embedding_function` parameter flows from main.py through embedder and collections
- `data_dir` means the same thing everywhere: the directory containing twin subdirectories
- `ZipValidationError` defined in importer.py, caught in ui.py
- `on_progress: Callable[[str], None] | None` — proper type annotation

### Review fixes applied:
- **Critical #1**: data_dir path convention standardized — importer writes directly to data_dir/<name>/ instead of relying on audit_facebook.save_outputs
- **Critical #2**: Explicit SentenceTransformerEmbeddingFunction passed to all ChromaDB operations
- **Critical #3**: twin_slug parameter added to create_chat_router, create_ui, chat_service — no more hardcoded "default"
- **Critical #4**: test_chat_truncates_long_message rewritten to properly mock LLM and verify truncation
- **Important #5**: Streaming deferred to Phase 1.1 (noted, not blocking MVP)
- **Important #6**: json.JSONDecodeError catch added to chat_service.py
- **Important #7**: chat_service.py extracted as shared layer — ui.py calls it, no duplication
- **Important #8**: Collection existence checked before overwrite, reported in result
- **Important #9**: sys.exit caught from audit_facebook.run_audit
- **Important #10**: GET /api/export endpoint added to chat.py
- **Important #11**: callable -> Callable[[str], None] | None
- **Suggestion #13**: SessionFactory pattern replaces shared session — thread-safe
- **Suggestion #16**: .env file creation added to Task 1
- **Suggestion #15**: find_inbox_folder checks 3 levels deep

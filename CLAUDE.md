# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Privacy-first AI digital twin platform. Import chat data (Facebook Messenger), build RAG-powered personality clones that respond in the twin's voice. Local-first: works with Ollama, OpenRouter, or OpenAI.

## Commands

```bash
# Run the app (FastAPI + Gradio UI on port 7860)
python -m app.main

# Run all tests
pytest

# Run a single test file
pytest tests/test_chat_service.py

# Run a single test
pytest tests/test_chat_service.py::test_function_name -v

# Docker
docker-compose up
```

## Architecture

**Request flow:** User message → FastAPI router (`app/chat.py`) or Gradio UI (`app/ui.py`) → `chat_service.chat()` → retrieval → LLM → persist to SQLite → response.

**Key modules:**
- `app/chat_service.py` — Core RAG pipeline: validation → `retrieve_chunks()` → prompt building → LLM call → DB persistence. Two modes: `answer` (normal RAG) and `rewrite` (voice cloning).
- `app/retrieval.py` — ChromaDB vector search with reranking. Scoring: embedding similarity * 0.6 + quality * 0.3 + DM boost * 0.1. Fetches 4x candidates, filters by quality (min 0.5), applies max_distance threshold (0.85).
- `app/importer.py` — Import pipeline: zip extraction → chunking via external scripts (`audit_facebook.py`, `score_and_chunk.py`) → embedding → ChromaDB storage.
- `app/adapters.py` — Format detection and E2EE message conversion.
- `app/sources.py` — Multi-source management per twin. Each source can be enabled/disabled independently.
- `app/embedder.py` — OpenAI-compatible embeddings wrapper (works with Ollama, OpenAI).
- `app/database.py` — SQLAlchemy models (ChatMessage, AppSetting) + SessionFactory.
- `app/config.py` — Settings dataclass. Three-level hierarchy: DB (AppSetting) > env vars (.env) > code defaults.
- `app/main.py` — FastAPI app creation, mounts Gradio UI.

**Data layout:**
```
data/<twin_slug>/
  sources.json              # Source manifest
  style_fingerprint.json    # Merged writing style stats
  sources/<source_id>/      # Per-source chunks and fingerprints
data/chromadb/              # Vector DB storage
db/chat_history.db          # SQLite chat history + settings
```

**Twin detection:** Scans `data/` for subdirectories containing `sources.json` or `style_fingerprint.json`. Auto-generates twin name from slug when `TWIN_NAME=auto`.

## AI Pipeline (when enriched metadata present)

The system has two code paths in `chat_service.chat()`:
- **Legacy path**: Direct RAG (cosine search → LLM). Used when chunks lack enriched metadata.
- **Pipeline path**: 4-agent LangGraph pipeline. Activates automatically when chunks have analyzer metadata.

**Pipeline flow:** Intent Agent → Context Agent (conditional) → Retriever Agent → Responder Agent → Critic Agent (with retry loop).

**Key modules:**
- `app/pipeline/graph.py` — LangGraph pipeline definition and `run_pipeline()` entry point.
- `app/pipeline/agents/` — One file per agent: intent, context, retriever, responder, critic.
- `app/pipeline/state.py` — `PipelineState` dataclass shared across agents.
- `app/pipeline/tone_map.py` — Tone similarity mapping for retrieval expansion.
- `app/pipeline/detect.py` — Checks if collection has enriched metadata.
- `app/analyzers/` — Metadata enrichment system: registry, stats, context, tone, emotion, persona.
- `app/chunking/` — Dynamic chunking: boundary detection + segment normalization.
- `app/backfill.py` — CLI to run missing analyzers on existing chunks.
- `app/rechunk.py` — CLI to re-chunk and re-analyze all data.

**Commands:**
```bash
python -m app.backfill                  # run missing analyzers on existing chunks
python -m app.backfill --analyzer tone_v1  # run specific analyzer
python -m app.rechunk                   # re-chunk + re-analyze all data
```

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review

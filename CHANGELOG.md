# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1.0] - 2026-04-03

### Added
- Pipeline detection system that automatically routes chat through the multi-agent LangGraph pipeline when enriched analyzer metadata is present, falling back to legacy RAG for unenriched collections
- Rechunk CLI (`python -m app.rechunk`) to re-process all raw chat data through dynamic boundary detection and analyzer enrichment, with safe temp-collection swap to prevent data loss
- Optional analyzer enrichment hook in the import pipeline for enriching chunks at import time
- UI now displays pipeline-specific metadata (intent, tone, retry count) when the pipeline path is active
- End-to-end integration tests covering the full pipeline flow including critic retry loop

### Changed
- `chat_service.chat()` restructured into `_legacy_chat` and `_pipeline_chat` with automatic routing based on collection metadata
- Error messages from the pipeline path are now sanitized to prevent leaking internal details to users

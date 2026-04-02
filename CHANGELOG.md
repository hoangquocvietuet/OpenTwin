# Changelog

All notable changes to this project will be documented in this file.

## [0.0.2.0] - 2026-04-03

### Fixed
- SSRF vulnerability in context agent: URL validation now blocks private/loopback/link-local IPs, non-http schemes, and DNS rebinding attacks (domain resolving to internal IPs)
- Context agent re-enables redirect following for legitimate URLs while validating the final destination
- Error leakage in responder agent: LLM exceptions no longer expose raw error details to users
- Critic agent now rejects on JSON parse failure (fail-safe) instead of silently approving bad output

### Changed
- `build_pipeline()` is now wired up with real agent functions via `functools.partial` instead of placeholder lambdas
- `run_pipeline()` uses the compiled LangGraph pipeline instead of manual sequential execution

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

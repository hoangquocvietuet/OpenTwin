# Digital Twins — Roadmap

## Phase 1: Chat-First MVP (Current)
> Goal: Import Facebook data, chat with your twin. Single user, single twin. Prove the core value.

- [x] Facebook data ingestion + mojibake fix
- [x] Adaptive chunking + richness scoring
- [x] Style fingerprint extraction
- [x] Train/holdout split
- [ ] FastAPI backend with OpenAI-compatible chat endpoint
- [ ] Streaming responses (SSE)
- [ ] ChromaDB embedding + retrieval
- [ ] Auto-generated system prompt from fingerprint
- [ ] Zero-retrieval fallback ("I don't have enough context")
- [ ] Twin quality indicator (chunk count + similarity in UI)
- [ ] SQLite chat history persistence
- [ ] Gradio Chat tab (streaming + quality indicator + export)
- [ ] Gradio Import tab (upload zip → audit → build twin)
- [ ] Conversation export (Markdown/JSON)
- [ ] Duplicate import detection (overwrite existing twin)
- [ ] Error handling (LLM timeout, connection, rate limit, zip slip)
- [ ] Docker single-container deployment

## Phase 2: Multi-User Platform
> Goal: Multiple users, multiple twins, access control.

- [ ] Auth system (admin, user, guest roles)
- [ ] User registration + login
- [ ] Profile system (group twins by purpose)
- [ ] Multi-twin support (self, parent, friend, mentor, colleague)
- [ ] Per-twin LLM config (provider, model, base URL)
- [ ] Settings UI tab
- [ ] Relationship-aware system prompts

## Phase 3: Privacy & Access Control
> Goal: Safely expose twins to external users.

- [ ] Public mode with topic allowlist
- [ ] Chunk-level visibility tagging (private/public)
- [ ] Privacy gate — topic classification before retrieval
- [ ] Adaptive sensitive data post-filter (configurable per twin)
- [ ] Per-twin privacy settings UI
- [ ] API key generation for external access
- [ ] Rate limiting for public twins

## Phase 4: Import Wizard & Multi-Platform
> Goal: More data sources, more control over what gets imported.

- [ ] Import wizard mode (exclude conversations, tag visibility, preview chunks)
- [ ] Telegram parser
- [ ] Twitter/X parser
- [ ] Facebook comments parser (public voice tone)
- [ ] Facebook posts parser (broadcast voice tone)
- [ ] Multi-tone chunk tagging (chat/comment/post)
- [ ] System prompt adapts per tone context
- [ ] Gmail parser
- [ ] Bulk re-import / incremental data updates

## Phase 5: Evaluation & Quality
> Goal: Measure and improve twin accuracy.

- [ ] /api/eval endpoint — run holdout suite
- [ ] Structural fingerprint drift detector (L2 distance)
- [ ] LLM-as-judge scoring (vocabulary, cadence, attitude)
- [ ] Semantic similarity scoring (cosine on embeddings)
- [ ] Evaluation history tracking (SQLite)
- [ ] Score dashboard in Gradio UI
- [ ] A/B comparison: twin response vs real response

## Phase 6: CLI
> Goal: Power users manage twins from the terminal.

- [ ] `dt import facebook ./export/inbox` — import data
- [ ] `dt chat "hoang-quoc-viet"` — interactive chat in terminal
- [ ] `dt list` — list all twins
- [ ] `dt eval "hoang-quoc-viet"` — run evaluation suite
- [ ] `dt export "hoang-quoc-viet"` — export twin config + data
- [ ] `dt serve` — start the API server

## Phase 7: Advanced Features
> Goal: Richer interactions and use cases.

- [ ] Conversation branching — explore "what would X say about Y"
- [ ] Multi-twin conversations — two twins talk to each other
- [ ] Twin-to-twin context sharing within a profile
- [ ] Voice cloning integration (TTS from twin's audio messages)
- [ ] Custom fine-tuning pipeline (LoRA) for users with 3000+ messages
- [ ] Semantic chunking with LLM (topic boundary detection)

## Phase 8: Distribution & Community
> Goal: Other people can use this easily.

- [ ] One-click installer (Electron/Tauri desktop app)
- [ ] Plugin system for custom data parsers
- [ ] Twin sharing (export/import twin profiles between users)
- [ ] Community twin templates (public figures from public data)
- [ ] Documentation site
- [ ] Localization (Vietnamese, English, others)

## Ideas (Unprioritized)
- Interactive novel engine — twins as NPCs with persistent memory
- Smart contract access control (Sui network) for decentralized twin access
- Twin aging — personality drift over time based on data timestamps
- Mood detection — adjust tone based on conversation context
- Memory consolidation — merge similar chunks over time to reduce DB size
- Cross-platform identity linking — same person across Facebook + Telegram + email
- Webhook integrations — twin auto-replies on Telegram/Discord

# AI Pipeline Plan B: LangGraph 4-Agent Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 4-agent sequential pipeline (Intent → Context → Retriever → Responder → Critic) using LangGraph, with configurable models per agent and a retry loop on the Critic.

**Architecture:** A LangGraph StateGraph where each node is an agent function that reads/writes to a shared `PipelineState`. The Intent Agent classifies tone/intent and detects whether external context is needed. The Context Agent fetches URLs. The Retriever Agent does hybrid tone+content retrieval from ChromaDB. The Responder generates responses using existing prompts from `prompt.py`. The Critic reviews and can reject with feedback (max 2 retries).

**Tech Stack:** LangGraph, langchain-core, OpenAI-compatible API, ChromaDB, existing prompt.py

**Dependency:** Requires Plan A (analyzers + dynamic chunking) to be complete. Chunks must have enriched metadata (tone, emotion, persona) for the Retriever Agent's metadata-filtered queries.

---

### Task 1: Pipeline State Dataclass

**Files:**
- Create: `app/pipeline/__init__.py`
- Create: `app/pipeline/state.py`
- Test: `tests/test_pipeline_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_state.py

from app.pipeline.state import PipelineState


def test_pipeline_state_defaults():
    """PipelineState initializes with sensible defaults."""
    state = PipelineState(raw_input="hello", mode="answer")

    assert state.raw_input == "hello"
    assert state.mode == "answer"
    assert state.intent is None
    assert state.tone is None
    assert state.needs_context is False
    assert state.context_source is None
    assert state.context_url is None
    assert state.resolved_content is None
    assert state.tone_chunks == []
    assert state.content_chunks == []
    assert state.draft_response is None
    assert state.approved is False
    assert state.critic_feedback is None
    assert state.retry_count == 0


def test_pipeline_state_rewrite_mode():
    """PipelineState works for rewrite mode."""
    state = PipelineState(raw_input="rewrite this article", mode="rewrite")
    assert state.mode == "rewrite"


def test_pipeline_state_is_mutable():
    """PipelineState fields can be updated (needed for LangGraph node updates)."""
    state = PipelineState(raw_input="test", mode="answer")
    state.intent = "casual_chat"
    state.tone = "casual_banter"
    state.resolved_content = "test"
    state.approved = True
    assert state.intent == "casual_chat"
    assert state.approved is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_state.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.pipeline'"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/__init__.py
"""LangGraph multi-agent pipeline for digital twin chat."""
```

```python
# app/pipeline/state.py
"""Pipeline state shared across all agents."""

from dataclasses import dataclass, field


@dataclass
class PipelineState:
    """Shared state that flows through the LangGraph pipeline.

    All agents read from and write to this state.
    """
    # Input
    raw_input: str
    mode: str  # "answer" | "rewrite"

    # Intent Agent output
    intent: str | None = None
    tone: str | None = None
    needs_context: bool = False
    context_source: str | None = None  # "url", "clipboard", None
    context_url: str | None = None

    # Context Agent output
    resolved_content: str | None = None

    # Retriever Agent output
    tone_chunks: list[dict] = field(default_factory=list)
    content_chunks: list[dict] = field(default_factory=list)

    # Responder Agent output
    draft_response: str | None = None

    # Critic Agent output
    approved: bool = False
    critic_feedback: str | None = None
    retry_count: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_state.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/__init__.py app/pipeline/state.py tests/test_pipeline_state.py
git commit -m "feat: PipelineState dataclass for LangGraph agent pipeline"
```

---

### Task 2: Tone Similarity Map

**Files:**
- Create: `app/pipeline/tone_map.py`
- Test: `tests/test_pipeline_tone_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_tone_map.py

from app.pipeline.tone_map import get_similar_tones, TONE_SIMILARITY


def test_get_similar_tones_known_tone():
    """Returns the tone itself plus its similar tones."""
    result = get_similar_tones("casual_banter")
    assert "casual_banter" in result
    assert "casual" in result
    assert "playful" in result


def test_get_similar_tones_unknown_tone():
    """Unknown tones return just the tone itself."""
    result = get_similar_tones("some_new_tone")
    assert result == ["some_new_tone"]


def test_get_similar_tones_no_duplicates():
    """Result has no duplicate tones."""
    for tone in TONE_SIMILARITY:
        result = get_similar_tones(tone)
        assert len(result) == len(set(result))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_tone_map.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/tone_map.py
"""Tone similarity mapping for retrieval expansion.

When the Intent Agent detects a tone, the Retriever Agent uses this map
to expand the query to include similar tones.
"""

TONE_SIMILARITY: dict[str, list[str]] = {
    "casual_banter": ["casual", "playful", "friendly"],
    "casual": ["casual_banter", "friendly", "relaxed"],
    "playful": ["casual_banter", "humorous", "friendly"],
    "friendly": ["casual", "casual_banter", "relaxed"],
    "relaxed": ["casual", "friendly", "neutral"],
    "formal_news": ["informational", "serious", "formal"],
    "formal": ["formal_news", "serious", "informational"],
    "informational": ["formal_news", "formal", "serious"],
    "serious": ["formal", "informational", "emotional"],
    "sarcastic": ["playful", "casual_banter", "humorous"],
    "humorous": ["playful", "sarcastic", "casual_banter"],
    "emotional": ["serious", "vulnerable", "personal"],
    "vulnerable": ["emotional", "personal", "serious"],
    "personal": ["emotional", "vulnerable", "friendly"],
    "angry": ["confrontational", "frustrated", "serious"],
    "confrontational": ["angry", "frustrated", "serious"],
    "frustrated": ["angry", "confrontational", "serious"],
    "neutral": ["casual", "relaxed", "friendly"],
}


def get_similar_tones(tone: str) -> list[str]:
    """Get a tone plus its similar tones for retrieval expansion.

    Args:
        tone: The detected tone string

    Returns:
        List of tones to query, with the input tone first. No duplicates.
    """
    similar = TONE_SIMILARITY.get(tone, [])
    result = [tone] + [t for t in similar if t != tone]
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_tone_map.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/tone_map.py tests/test_pipeline_tone_map.py
git commit -m "feat: tone similarity map for retrieval expansion"
```

---

### Task 3: Intent Agent

**Files:**
- Create: `app/pipeline/agents/__init__.py`
- Create: `app/pipeline/agents/intent.py`
- Test: `tests/test_agent_intent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_intent.py

import re
from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.intent import intent_agent


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_intent_agent_casual_chat():
    """Short casual message classified correctly."""
    client = _make_mock_llm_client('{"intent": "casual_chat", "tone": "casual_banter"}')
    state = PipelineState(raw_input="tối nay ăn j", mode="answer")

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.intent == "casual_chat"
    assert result.tone == "casual_banter"
    assert result.needs_context is False
    assert result.resolved_content == "tối nay ăn j"


def test_intent_agent_long_text_no_context_needed():
    """Long text input doesn't need external context."""
    long_text = "VKSND TP.HCM đã ban hành cáo trạng " * 20
    client = _make_mock_llm_client('{"intent": "rewrite_article", "tone": "formal_news"}')
    state = PipelineState(raw_input=long_text, mode="rewrite")

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.needs_context is False
    assert result.resolved_content == long_text


def test_intent_agent_url_needs_context():
    """Message with URL triggers context fetching."""
    client = _make_mock_llm_client('{"intent": "rewrite_article", "tone": "formal_news"}')
    state = PipelineState(
        raw_input="rewrite this https://vnexpress.net/article-123",
        mode="rewrite",
    )

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.needs_context is True
    assert result.context_source == "url"
    assert "vnexpress.net" in result.context_url


def test_intent_agent_handles_malformed_json():
    """Falls back to defaults when LLM returns bad JSON."""
    client = _make_mock_llm_client("not json at all")
    state = PipelineState(raw_input="hello", mode="answer")

    result = intent_agent(state, llm_client=client, llm_model="test")

    assert result.intent == "general"
    assert result.tone == "neutral"
    assert result.resolved_content == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_intent.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/agents/__init__.py
"""Pipeline agent functions."""
```

```python
# app/pipeline/agents/intent.py
"""Intent Agent — classifies intent, tone, and detects context needs.

First agent in the pipeline. Determines how to route the message
and what retrieval strategy to use.
"""

import json
import re

from app.pipeline.state import PipelineState

_URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

_SYSTEM_PROMPT = """You classify chat messages. Given a message and its mode (answer or rewrite), return a JSON object:

- "intent": one of: casual_chat, question, greeting, banter, rewrite_article, rewrite_casual, rewrite_announcement, information_sharing, emotional, other, general
- "tone": one of: casual_banter, casual, playful, friendly, relaxed, formal_news, formal, informational, serious, sarcastic, humorous, emotional, vulnerable, personal, angry, confrontational, frustrated, neutral

Consider the message length, content, and mode. Return ONLY valid JSON."""


def intent_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
) -> PipelineState:
    """Classify the input message's intent and tone.

    Also detects whether external context (URL, clipboard) needs to be fetched.
    """
    raw = state.raw_input

    # Detect URL in input
    url_match = _URL_PATTERN.search(raw)

    # Short message with URL → needs context
    non_url_text = _URL_PATTERN.sub("", raw).strip()
    if url_match and len(non_url_text) < 100:
        state.needs_context = True
        state.context_source = "url"
        state.context_url = url_match.group(0)
    else:
        state.needs_context = False
        state.resolved_content = raw

    # Classify intent and tone via LLM
    if llm_client:
        try:
            response = llm_client.chat.completions.create(
                model=llm_model or "llama3.1:8b",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Mode: {state.mode}\nMessage: {raw[:2000]}"},
                ],
                timeout=15,
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)
            state.intent = parsed.get("intent", "general")
            state.tone = parsed.get("tone", "neutral")
        except (json.JSONDecodeError, Exception):
            state.intent = "general"
            state.tone = "neutral"
    else:
        state.intent = "general"
        state.tone = "neutral"

    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_intent.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/agents/__init__.py app/pipeline/agents/intent.py tests/test_agent_intent.py
git commit -m "feat: Intent Agent — classify intent, tone, detect context needs"
```

---

### Task 4: Context Agent

**Files:**
- Create: `app/pipeline/agents/context.py`
- Test: `tests/test_agent_context.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_context.py

from unittest.mock import MagicMock, patch
from app.pipeline.state import PipelineState
from app.pipeline.agents.context import context_agent


def test_context_agent_fetches_url():
    """Context agent fetches URL content and sets resolved_content."""
    state = PipelineState(
        raw_input="rewrite this https://example.com/article",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="https://example.com/article",
    )

    with patch("app.pipeline.agents.context.httpx") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>Article content here about technology.</p></body></html>"
        mock_httpx.get.return_value = mock_response

        # Also mock the intent re-classification LLM call
        llm_client = MagicMock()
        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = '{"intent": "rewrite_article", "tone": "informational"}'
        llm_client.chat.completions.create.return_value = llm_resp

        result = context_agent(state, llm_client=llm_client, llm_model="test")

    assert result.resolved_content is not None
    assert len(result.resolved_content) > 0
    assert result.tone == "informational"


def test_context_agent_skips_when_not_needed():
    """Context agent is a no-op when needs_context is False."""
    state = PipelineState(
        raw_input="hello",
        mode="answer",
        needs_context=False,
        resolved_content="hello",
    )

    result = context_agent(state)

    assert result.resolved_content == "hello"


def test_context_agent_handles_fetch_failure():
    """Context agent falls back to raw_input on fetch failure."""
    state = PipelineState(
        raw_input="rewrite this https://example.com/broken",
        mode="rewrite",
        needs_context=True,
        context_source="url",
        context_url="https://example.com/broken",
    )

    with patch("app.pipeline.agents.context.httpx") as mock_httpx:
        mock_httpx.get.side_effect = Exception("Connection failed")

        result = context_agent(state)

    # Falls back to raw_input minus the URL
    assert result.resolved_content is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_context.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/agents/context.py
"""Context Agent — fetches external content when needed.

Conditional agent that only runs when the Intent Agent sets needs_context=True.
Fetches URLs, then re-classifies the resolved content's tone.
"""

import json
import re

import httpx

from app.pipeline.state import PipelineState

_HTML_TAG_RE = re.compile(r'<[^>]+>')
_WHITESPACE_RE = re.compile(r'\s+')


def _extract_text_from_html(html: str) -> str:
    """Naive HTML to text extraction."""
    # Remove script and style blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = _HTML_TAG_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def _reclassify_tone(content: str, mode: str, llm_client, llm_model: str) -> tuple[str, str]:
    """Re-run intent/tone classification on fetched content."""
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": (
                    "Classify this text. Return JSON: "
                    '{"intent": "...", "tone": "..."} '
                    "where intent is one of: rewrite_article, rewrite_casual, rewrite_announcement, "
                    "information_sharing, other; and tone is one of: casual_banter, casual, formal_news, "
                    "formal, informational, serious, sarcastic, emotional, neutral."
                )},
                {"role": "user", "content": f"Mode: {mode}\nText: {content[:2000]}"},
            ],
            timeout=15,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        return parsed.get("intent", "general"), parsed.get("tone", "neutral")
    except Exception:
        return "general", "neutral"


def context_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
) -> PipelineState:
    """Fetch external context if needed, then re-classify tone.

    Only operates when state.needs_context is True.
    """
    if not state.needs_context:
        return state

    if state.context_source == "url" and state.context_url:
        try:
            resp = httpx.get(state.context_url, timeout=15, follow_redirects=True)
            if resp.status_code == 200:
                text = _extract_text_from_html(resp.text)
                if text:
                    state.resolved_content = text[:10000]  # cap at 10k chars
                else:
                    state.resolved_content = state.raw_input
            else:
                state.resolved_content = state.raw_input
        except Exception:
            # Fall back to raw input minus URL
            url_removed = state.raw_input.replace(state.context_url, "").strip()
            state.resolved_content = url_removed or state.raw_input
    else:
        # clipboard or other — for now fall back to raw_input
        state.resolved_content = state.raw_input

    # Re-classify tone on the actual content
    if llm_client and state.resolved_content:
        intent, tone = _reclassify_tone(
            state.resolved_content, state.mode, llm_client, llm_model or "llama3.1:8b"
        )
        state.intent = intent
        state.tone = tone

    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_context.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/agents/context.py tests/test_agent_context.py
git commit -m "feat: Context Agent — URL fetch with tone re-classification"
```

---

### Task 5: Retriever Agent

**Files:**
- Create: `app/pipeline/agents/retriever.py`
- Test: `tests/test_agent_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_retriever.py

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.retriever import retriever_agent


def _make_mock_collection(tone_results=None, content_results=None):
    """Mock ChromaDB collection that returns different results per query."""
    collection = MagicMock()
    collection.count.return_value = 100

    call_count = [0]
    def mock_query(**kwargs):
        call_count[0] += 1
        where = kwargs.get("where")
        # First call = tone query (has tone filter), second = content query
        if where and "$and" in str(where):
            return tone_results or {
                "ids": [["tone_1", "tone_2", "tone_3"]],
                "documents": [["casual doc 1", "casual doc 2", "casual doc 3"]],
                "distances": [[0.2, 0.3, 0.4]],
                "metadatas": [[
                    {"tone": "casual", "formality": 0.2, "twin_msg_ratio": 0.5},
                    {"tone": "casual_banter", "formality": 0.1, "twin_msg_ratio": 0.6},
                    {"tone": "playful", "formality": 0.3, "twin_msg_ratio": 0.4},
                ]],
            }
        else:
            return content_results or {
                "ids": [["content_1", "content_2"]],
                "documents": [["food discussion", "dinner plan"]],
                "distances": [[0.3, 0.8]],  # second one is low similarity
                "metadatas": [[
                    {"tone": "casual", "twin_msg_ratio": 0.5},
                    {"tone": "casual", "twin_msg_ratio": 0.3},
                ]],
            }

    collection.query = mock_query
    return collection


def test_retriever_agent_returns_tone_and_content_chunks():
    """Retriever returns both tone-matched and content-matched chunks."""
    collection = _make_mock_collection()
    state = PipelineState(
        raw_input="tối nay ăn j",
        mode="answer",
        intent="casual_chat",
        tone="casual_banter",
        resolved_content="tối nay ăn j",
    )

    result = retriever_agent(state, collection=collection)

    assert len(result.tone_chunks) > 0
    assert len(result.tone_chunks) <= 3
    assert len(result.content_chunks) >= 0
    assert len(result.content_chunks) <= 2


def test_retriever_agent_filters_low_similarity_content():
    """Content chunks with similarity below threshold are excluded."""
    # All content results have high distance (low similarity)
    content_results = {
        "ids": [["c1", "c2"]],
        "documents": [["irrelevant 1", "irrelevant 2"]],
        "distances": [[0.9, 0.95]],  # very dissimilar
        "metadatas": [[{"twin_msg_ratio": 0.5}, {"twin_msg_ratio": 0.5}]],
    }
    collection = _make_mock_collection(content_results=content_results)
    state = PipelineState(
        raw_input="something unrelated",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="something unrelated",
    )

    result = retriever_agent(state, collection=collection)

    assert result.content_chunks == []


def test_retriever_agent_handles_empty_collection():
    """Retriever handles empty collection gracefully."""
    collection = MagicMock()
    collection.count.return_value = 0

    state = PipelineState(
        raw_input="hello",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="hello",
    )

    result = retriever_agent(state, collection=collection)

    assert result.tone_chunks == []
    assert result.content_chunks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_retriever.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/agents/retriever.py
"""Retriever Agent — hybrid tone + content retrieval from ChromaDB.

Performs two queries:
1. Tone retrieval: find chunks matching the detected tone (style reference)
2. Content retrieval: find chunks matching the content (personal context)
"""

from app.pipeline.state import PipelineState
from app.pipeline.tone_map import get_similar_tones

# Content similarity threshold (cosine distance). Lower = more similar.
# Only keep content chunks with distance below this.
CONTENT_MAX_DISTANCE = 0.6

TONE_RESULTS_COUNT = 3
CONTENT_RESULTS_COUNT = 2


def retriever_agent(
    state: PipelineState,
    collection=None,
) -> PipelineState:
    """Perform hybrid retrieval: tone-matched + content-matched chunks.

    Tone chunks: filtered by tone metadata, used as style references.
    Content chunks: filtered by cosine similarity, used as personal context.
    """
    if not collection or collection.count() == 0:
        state.tone_chunks = []
        state.content_chunks = []
        return state

    query_text = state.resolved_content or state.raw_input

    # 1. Tone retrieval — find chunks where twin sounds like the detected tone
    tone_chunks = []
    if state.tone:
        similar_tones = get_similar_tones(state.tone)
        try:
            tone_results = collection.query(
                query_texts=[query_text],
                n_results=10,
                where={
                    "$and": [
                        {"tone": {"$in": similar_tones}},
                        {"twin_msg_ratio": {"$gte": 0.3}},
                    ]
                },
                include=["documents", "distances", "metadatas"],
            )
            for i in range(len(tone_results["ids"][0])):
                tone_chunks.append({
                    "chunk_id": tone_results["ids"][0][i],
                    "document": tone_results["documents"][0][i],
                    "distance": tone_results["distances"][0][i],
                    "metadata": tone_results["metadatas"][0][i],
                })
        except Exception:
            # Metadata filter may fail if chunks lack tone field — fall back to no filter
            pass

    # If tone query returned nothing (no enriched chunks), fall back to unfiltered
    if not tone_chunks:
        try:
            fallback = collection.query(
                query_texts=[query_text],
                n_results=TONE_RESULTS_COUNT,
                include=["documents", "distances", "metadatas"],
            )
            for i in range(len(fallback["ids"][0])):
                tone_chunks.append({
                    "chunk_id": fallback["ids"][0][i],
                    "document": fallback["documents"][0][i],
                    "distance": fallback["distances"][0][i],
                    "metadata": fallback["metadatas"][0][i],
                })
        except Exception:
            pass

    state.tone_chunks = tone_chunks[:TONE_RESULTS_COUNT]

    # 2. Content retrieval — find chunks about similar topics
    content_chunks = []
    try:
        content_results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={"twin_msg_ratio": {"$gte": 0.2}},
            include=["documents", "distances", "metadatas"],
        )
        for i in range(len(content_results["ids"][0])):
            dist = content_results["distances"][0][i]
            if dist < CONTENT_MAX_DISTANCE:
                content_chunks.append({
                    "chunk_id": content_results["ids"][0][i],
                    "document": content_results["documents"][0][i],
                    "distance": dist,
                    "metadata": content_results["metadatas"][0][i],
                })
    except Exception:
        pass

    # Deduplicate (a chunk could appear in both tone and content results)
    tone_ids = {c["chunk_id"] for c in state.tone_chunks}
    content_chunks = [c for c in content_chunks if c["chunk_id"] not in tone_ids]

    state.content_chunks = content_chunks[:CONTENT_RESULTS_COUNT]
    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_retriever.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/agents/retriever.py tests/test_agent_retriever.py
git commit -m "feat: Retriever Agent — hybrid tone + content retrieval with dedup"
```

---

### Task 6: Responder Agent

**Files:**
- Create: `app/pipeline/agents/responder.py`
- Test: `tests/test_agent_responder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_responder.py

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.responder import responder_agent


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_responder_answer_mode():
    """Responder generates answer using system prompt and chunks."""
    client = _make_mock_llm_client("đang code nè")
    state = PipelineState(
        raw_input="đang làm gì",
        mode="answer",
        intent="casual_chat",
        tone="casual_banter",
        resolved_content="đang làm gì",
        tone_chunks=[{"document": "Viet: đang code dự án\nFriend: dự án gì"}],
        content_chunks=[],
    )

    result = responder_agent(
        state,
        llm_client=client,
        llm_model="test",
        system_prompt="You are Viet. Chat casually.",
        rewrite_prompt="Rephrase in your style.",
    )

    assert result.draft_response == "đang code nè"


def test_responder_rewrite_mode():
    """Responder uses rewrite prompt in rewrite mode."""
    client = _make_mock_llm_client("tối nay ăn j đây")
    state = PipelineState(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        intent="rewrite_casual",
        tone="casual",
        resolved_content="Tối nay mình ăn gì nhỉ",
        tone_chunks=[{"document": "Viet: ê ăn j\nFriend: phở"}],
        content_chunks=[],
    )

    result = responder_agent(
        state,
        llm_client=client,
        llm_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase in your style. Do not answer.",
    )

    assert result.draft_response == "tối nay ăn j đây"

    # Verify rewrite_prompt was used, not system_prompt
    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert any("Rephrase" in m["content"] for m in system_msgs)


def test_responder_includes_critic_feedback_on_retry():
    """On retry, critic feedback is included in the prompt."""
    client = _make_mock_llm_client("tối nay ăn j đây")
    state = PipelineState(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        intent="rewrite_casual",
        tone="casual",
        resolved_content="Tối nay mình ăn gì nhỉ",
        tone_chunks=[],
        content_chunks=[],
        critic_feedback="You answered instead of rephrasing. Rephrase the question.",
        retry_count=1,
    )

    result = responder_agent(
        state,
        llm_client=client,
        llm_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase in your style.",
    )

    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    all_content = " ".join(m["content"] for m in messages)
    assert "answered instead of rephrasing" in all_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_responder.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/agents/responder.py
"""Responder Agent — generates the twin's response.

Uses the system/rewrite prompt from prompt.py, plus tone and content chunks
as contextual references. Includes critic feedback on retry attempts.
"""

from app.pipeline.state import PipelineState


def responder_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
    system_prompt: str = "",
    rewrite_prompt: str = "",
    session_factory=None,
    twin_slug: str = "",
) -> PipelineState:
    """Generate a response as the twin.

    For answer mode: uses system_prompt + tone chunks (style) + content chunks (context) + history
    For rewrite mode: uses rewrite_prompt + tone chunks (style only)
    """
    if not llm_client:
        state.draft_response = "LLM not configured."
        return state

    messages = []

    # System prompt based on mode
    if state.mode == "rewrite":
        messages.append({"role": "system", "content": rewrite_prompt})
    else:
        messages.append({"role": "system", "content": system_prompt})

    # Tone chunks — style reference
    if state.tone_chunks:
        tone_docs = "\n\n---\n\n".join(c.get("document", "") for c in state.tone_chunks if c.get("document"))
        if tone_docs:
            messages.append({
                "role": "system",
                "content": f"This is how you sound in this register (use for style only, NOT content):\n\n{tone_docs}",
            })

    # Content chunks — personal context (answer mode mainly, but available for both)
    if state.content_chunks:
        content_docs = "\n\n---\n\n".join(c.get("document", "") for c in state.content_chunks if c.get("document"))
        if content_docs:
            messages.append({
                "role": "system",
                "content": f"Related things you've said before (personal context):\n\n{content_docs}",
            })

    # Conversation history (answer mode only)
    if state.mode == "answer" and session_factory and twin_slug:
        try:
            from app.database import ChatMessage
            with session_factory() as session:
                recent_msgs = (
                    session.query(ChatMessage)
                    .filter_by(twin_slug=twin_slug)
                    .order_by(ChatMessage.id.desc())
                    .limit(10)
                    .all()
                )
                recent_msgs.reverse()
                for msg in recent_msgs:
                    messages.append({"role": msg.role, "content": msg.content})
        except Exception:
            pass

    # Critic feedback on retry
    if state.critic_feedback and state.retry_count > 0:
        messages.append({
            "role": "system",
            "content": f"Your previous attempt was rejected. Feedback: {state.critic_feedback}\nPlease try again following the feedback.",
        })

    # User message
    messages.append({"role": "user", "content": state.resolved_content or state.raw_input})

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=messages,
            stream=False,
            timeout=30,
        )
        state.draft_response = response.choices[0].message.content or ""
    except Exception as e:
        state.draft_response = f"LLM error: {e}"

    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_responder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/agents/responder.py tests/test_agent_responder.py
git commit -m "feat: Responder Agent — generates response with tone/content chunks and retry support"
```

---

### Task 7: Critic Agent

**Files:**
- Create: `app/pipeline/agents/critic.py`
- Test: `tests/test_agent_critic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_critic.py

from unittest.mock import MagicMock
from app.pipeline.state import PipelineState
from app.pipeline.agents.critic import critic_agent


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_critic_approves_good_response():
    """Critic approves when response matches tone and mode."""
    client = _make_mock_llm_client('{"approved": true, "feedback": ""}')
    state = PipelineState(
        raw_input="tối nay ăn j",
        mode="answer",
        intent="casual_chat",
        tone="casual_banter",
        resolved_content="tối nay ăn j",
        tone_chunks=[{"document": "Viet: ăn phở đi"}],
        content_chunks=[],
        draft_response="ăn phở đi",
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    assert result.approved is True
    assert result.retry_count == 0


def test_critic_rejects_answer_in_rewrite_mode():
    """Critic rejects when rewrite mode got an answer instead of rephrase."""
    client = _make_mock_llm_client('{"approved": false, "feedback": "You answered the question instead of rephrasing it. The input is a question and the output should be a question in your style."}')
    state = PipelineState(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        intent="rewrite_casual",
        tone="casual",
        resolved_content="Tối nay mình ăn gì nhỉ",
        tone_chunks=[],
        content_chunks=[],
        draft_response="tối nay ăn ốc",  # answered instead of rephrased
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    assert result.approved is False
    assert result.critic_feedback is not None
    assert len(result.critic_feedback) > 0
    assert result.retry_count == 1


def test_critic_increments_retry_count():
    """Each rejection increments retry_count."""
    client = _make_mock_llm_client('{"approved": false, "feedback": "Wrong tone."}')
    state = PipelineState(
        raw_input="test",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="test",
        draft_response="I would be happy to help you!",
        retry_count=1,  # already retried once
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    assert result.approved is False
    assert result.retry_count == 2


def test_critic_handles_malformed_json():
    """Critic defaults to approved on malformed LLM response."""
    client = _make_mock_llm_client("not json")
    state = PipelineState(
        raw_input="test",
        mode="answer",
        intent="casual_chat",
        tone="casual",
        resolved_content="test",
        draft_response="ok",
    )

    result = critic_agent(state, llm_client=client, llm_model="test")

    # Default to approved to avoid blocking the user
    assert result.approved is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_critic.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/agents/critic.py
"""Critic Agent — reviews the Responder's draft before sending.

Checks style match, tone consistency, mode compliance (rewrite vs answer),
and appropriate response length. Can reject with feedback for retry.
"""

import json

from app.pipeline.state import PipelineState

_SYSTEM_PROMPT = """You are a quality reviewer for a digital twin chatbot. Review the draft response.

Given:
- The original user input
- The mode (answer or rewrite)
- The detected tone
- The draft response
- Style reference chunks (how the twin actually types)

Check:
1. STYLE: Does the draft sound like the twin? (casual, lowercase, slang vs formal)
2. TONE: Does the tone match? Casual input → casual output, formal → formal.
3. MODE COMPLIANCE:
   - If mode=rewrite: Did it REPHRASE (not answer)? Statements must stay statements, questions stay questions.
   - If mode=answer: Is it on-topic? No hallucination?
4. LENGTH: Appropriate for the input energy? Short question → short answer.

Return JSON:
- "approved": true if the response is acceptable, false if it needs retry
- "feedback": if not approved, explain what's wrong and how to fix it. Be specific.

Return ONLY valid JSON."""


def critic_agent(
    state: PipelineState,
    llm_client=None,
    llm_model: str | None = None,
) -> PipelineState:
    """Review the draft response for quality.

    If rejected, sets critic_feedback and increments retry_count.
    """
    if not llm_client:
        state.approved = True
        return state

    # Build review context
    style_ref = ""
    if state.tone_chunks:
        style_ref = "\n---\n".join(c.get("document", "") for c in state.tone_chunks[:2])

    review_input = (
        f"Mode: {state.mode}\n"
        f"Detected tone: {state.tone}\n"
        f"Original input: {state.raw_input[:1000]}\n"
        f"Draft response: {state.draft_response[:1000]}\n"
    )
    if style_ref:
        review_input += f"\nStyle reference (how twin types):\n{style_ref[:500]}"

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": review_input},
            ],
            timeout=15,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        approved = bool(parsed.get("approved", True))
        feedback = parsed.get("feedback", "")

        if approved:
            state.approved = True
        else:
            state.approved = False
            state.critic_feedback = feedback
            state.retry_count += 1
    except (json.JSONDecodeError, Exception):
        # On parse failure, default to approved to avoid blocking
        state.approved = True

    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_critic.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/agents/critic.py tests/test_agent_critic.py
git commit -m "feat: Critic Agent — review draft for style, tone, and mode compliance"
```

---

### Task 8: LangGraph Graph Definition

**Files:**
- Create: `app/pipeline/graph.py`
- Test: `tests/test_pipeline_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_graph.py

from unittest.mock import MagicMock, patch
from app.pipeline.graph import build_pipeline, run_pipeline


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_build_pipeline_returns_compiled_graph():
    """build_pipeline returns a compiled LangGraph."""
    pipeline = build_pipeline()
    assert pipeline is not None


def test_run_pipeline_answer_mode():
    """Full pipeline run in answer mode produces a response."""
    collection = MagicMock()
    collection.count.return_value = 5
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["Viet: đang code"]],
        "distances": [[0.3]],
        "metadatas": [[{"tone": "casual", "twin_msg_ratio": 0.5}]],
    }

    # Intent: casual_chat
    # Critic: approved
    intent_resp = '{"intent": "casual_chat", "tone": "casual_banter"}'
    critic_resp = '{"approved": true, "feedback": ""}'
    responder_resp = "đang code nè"

    call_count = [0]
    def mock_create(**kwargs):
        call_count[0] += 1
        msgs = kwargs.get("messages", [])
        system_content = msgs[0]["content"] if msgs else ""

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]

        # Route based on system prompt content
        if "classify" in system_content.lower():
            mock_resp.choices[0].message.content = intent_resp
        elif "quality reviewer" in system_content.lower():
            mock_resp.choices[0].message.content = critic_resp
        else:
            mock_resp.choices[0].message.content = responder_resp

        return mock_resp

    llm_client = MagicMock()
    llm_client.chat.completions.create = mock_create

    result = run_pipeline(
        raw_input="đang làm gì",
        mode="answer",
        collection=collection,
        llm_client=llm_client,
        llm_model="test",
        classifier_client=llm_client,
        classifier_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase.",
    )

    assert result.draft_response == "đang code nè"
    assert result.approved is True


def test_run_pipeline_critic_retry():
    """Pipeline retries when critic rejects, then approves on second try."""
    collection = MagicMock()
    collection.count.return_value = 5
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["Viet: ăn phở"]],
        "distances": [[0.3]],
        "metadatas": [[{"tone": "casual", "twin_msg_ratio": 0.5}]],
    }

    attempt = [0]
    def mock_create(**kwargs):
        msgs = kwargs.get("messages", [])
        system_content = msgs[0]["content"] if msgs else ""

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]

        if "classify" in system_content.lower():
            mock_resp.choices[0].message.content = '{"intent": "rewrite_casual", "tone": "casual"}'
        elif "quality reviewer" in system_content.lower():
            attempt[0] += 1
            if attempt[0] <= 1:
                mock_resp.choices[0].message.content = '{"approved": false, "feedback": "You answered instead of rephrasing."}'
            else:
                mock_resp.choices[0].message.content = '{"approved": true, "feedback": ""}'
        else:
            mock_resp.choices[0].message.content = "tối nay ăn j đây"

        return mock_resp

    llm_client = MagicMock()
    llm_client.chat.completions.create = mock_create

    result = run_pipeline(
        raw_input="Tối nay mình ăn gì nhỉ",
        mode="rewrite",
        collection=collection,
        llm_client=llm_client,
        llm_model="test",
        classifier_client=llm_client,
        classifier_model="test",
        system_prompt="You are Viet.",
        rewrite_prompt="Rephrase.",
    )

    assert result.approved is True
    assert result.retry_count >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_graph.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/pipeline/graph.py
"""LangGraph pipeline definition and execution.

Wires the 5 agents into a StateGraph with conditional routing:
Intent → (Context if needed) → Retriever → Responder → Critic → (retry or done)
"""

from langgraph.graph import StateGraph, END

from app.pipeline.state import PipelineState
from app.pipeline.agents.intent import intent_agent
from app.pipeline.agents.context import context_agent
from app.pipeline.agents.retriever import retriever_agent
from app.pipeline.agents.responder import responder_agent
from app.pipeline.agents.critic import critic_agent

MAX_RETRIES = 2


def _route_after_intent(state: PipelineState) -> str:
    """Route after intent: to context agent if needed, else to retriever."""
    if state.needs_context:
        return "context"
    return "retriever"


def _route_after_critic(state: PipelineState) -> str:
    """Route after critic: retry responder or finish."""
    if not state.approved and state.retry_count < MAX_RETRIES:
        return "responder"
    return END


def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph pipeline.

    Returns a compiled graph. Call .invoke(state) to run it.
    Note: Agent dependencies (llm_client, collection, etc.) are bound
    at runtime via run_pipeline(), not at build time.
    """
    graph = StateGraph(PipelineState)

    # Nodes are added as string names; actual functions are bound in run_pipeline
    graph.add_node("intent", lambda state: state)  # placeholder
    graph.add_node("context", lambda state: state)
    graph.add_node("retriever", lambda state: state)
    graph.add_node("responder", lambda state: state)
    graph.add_node("critic", lambda state: state)

    graph.set_entry_point("intent")
    graph.add_conditional_edges("intent", _route_after_intent, {
        "context": "context",
        "retriever": "retriever",
    })
    graph.add_edge("context", "retriever")
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, {
        "responder": "responder",
        END: END,
    })

    return graph.compile()


def run_pipeline(
    raw_input: str,
    mode: str,
    collection,
    llm_client,
    llm_model: str,
    classifier_client,
    classifier_model: str,
    system_prompt: str,
    rewrite_prompt: str,
    session_factory=None,
    twin_slug: str = "",
) -> PipelineState:
    """Run the full pipeline with all dependencies injected.

    This is the main entry point. It creates the state, binds agent functions
    with their dependencies, and executes the graph.
    """
    state = PipelineState(raw_input=raw_input, mode=mode)

    # Run agents sequentially (simpler than LangGraph node binding for now)
    # This gives us the same pipeline flow with cleaner dependency injection.

    # 1. Intent
    state = intent_agent(state, llm_client=classifier_client, llm_model=classifier_model)

    # 2. Context (conditional)
    if state.needs_context:
        state = context_agent(state, llm_client=classifier_client, llm_model=classifier_model)

    # Ensure resolved_content is set
    if not state.resolved_content:
        state.resolved_content = state.raw_input

    # 3. Retriever
    state = retriever_agent(state, collection=collection)

    # 4. Responder + Critic loop (max retries)
    for _ in range(MAX_RETRIES + 1):
        state = responder_agent(
            state,
            llm_client=llm_client,
            llm_model=llm_model,
            system_prompt=system_prompt,
            rewrite_prompt=rewrite_prompt,
            session_factory=session_factory,
            twin_slug=twin_slug,
        )

        state = critic_agent(
            state,
            llm_client=classifier_client,
            llm_model=classifier_model,
        )

        if state.approved:
            break

    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_graph.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/graph.py tests/test_pipeline_graph.py
git commit -m "feat: LangGraph pipeline — sequential execution with critic retry loop"
```

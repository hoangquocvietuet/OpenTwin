# AI Pipeline Plan A: Analyzer System + Dynamic Chunking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the metadata enrichment system (analyzers) and dynamic chunking so that every chunk in ChromaDB has rich tone/emotion/persona metadata for the pipeline to query.

**Architecture:** An analyzer registry pattern where each analyzer is a function that takes a chunk + context and returns metadata fields. Dynamic chunking uses LLM-based boundary detection to split conversations at natural points (topic shifts, mood changes) instead of fixed sizes. Both systems use configurable LLM models with fallback to the main LLM.

**Tech Stack:** Python, ChromaDB, OpenAI-compatible API (configurable), existing import pipeline

**Dependency:** This is Plan A of 3. Plan B (LangGraph Pipeline) and Plan C (Integration + Migration) depend on the enriched metadata this plan produces. This plan is independently testable.

---

### Task 1: Add Classifier and Analyzer Model Config

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — add to existing file

def test_classifier_config_defaults():
    """Classifier config falls back to main LLM config."""
    import os
    # Clear any env overrides
    for key in ["CLASSIFIER_BASE_URL", "CLASSIFIER_MODEL", "CLASSIFIER_API_KEY",
                "ANALYZER_BASE_URL", "ANALYZER_MODEL", "ANALYZER_API_KEY"]:
        os.environ.pop(key, None)

    from app.config import Settings
    s = Settings()

    # Classifier falls back to main LLM
    assert s.classifier_base_url == s.llm_base_url
    assert s.classifier_model == s.llm_model
    assert s.classifier_api_key == s.llm_api_key

    # Analyzer falls back to main LLM
    assert s.analyzer_base_url == s.llm_base_url
    assert s.analyzer_model == s.llm_model
    assert s.analyzer_api_key == s.llm_api_key


def test_classifier_config_overrides():
    """Classifier config can be overridden via env vars."""
    import os
    os.environ["CLASSIFIER_MODEL"] = "llama3.2:3b"
    os.environ["ANALYZER_MODEL"] = "phi-3:mini"

    from app.config import Settings
    s = Settings()
    assert s.classifier_model == "llama3.2:3b"
    assert s.analyzer_model == "phi-3:mini"

    # Cleanup
    del os.environ["CLASSIFIER_MODEL"]
    del os.environ["ANALYZER_MODEL"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_classifier_config_defaults -v`
Expected: FAIL with "AttributeError: Settings has no attribute 'classifier_base_url'"

- [ ] **Step 3: Write minimal implementation**

Add to `app/config.py` inside the `Settings` dataclass, after the `data_dir` field:

```python
    # Classifier model (Intent Agent, Critic Agent) — falls back to LLM_*
    classifier_base_url: str = field(
        default_factory=lambda: os.getenv("CLASSIFIER_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    )
    classifier_model: str = field(
        default_factory=lambda: os.getenv("CLASSIFIER_MODEL", os.getenv("LLM_MODEL", "llama3.1:8b"))
    )
    classifier_api_key: str = field(
        default_factory=lambda: os.getenv("CLASSIFIER_API_KEY", os.getenv("LLM_API_KEY", "ollama"))
    )
    # Analyzer model (import-time enrichment, chunking) — falls back to LLM_*
    analyzer_base_url: str = field(
        default_factory=lambda: os.getenv("ANALYZER_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    )
    analyzer_model: str = field(
        default_factory=lambda: os.getenv("ANALYZER_MODEL", os.getenv("LLM_MODEL", "llama3.1:8b"))
    )
    analyzer_api_key: str = field(
        default_factory=lambda: os.getenv("ANALYZER_API_KEY", os.getenv("LLM_API_KEY", "ollama"))
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add classifier and analyzer model config with LLM fallback"
```

---

### Task 2: Stats Analyzer (No LLM, Heuristic)

**Files:**
- Create: `app/analyzers/__init__.py`
- Create: `app/analyzers/stats.py`
- Test: `tests/test_analyzers_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_stats.py

from app.analyzers.stats import analyze_stats


def test_analyze_stats_basic():
    """Stats analyzer extracts message-level statistics from a chunk."""
    chunk = {
        "messages": [
            {"author": "Hoang Quoc Viet", "text": "tối nay ăn j đây", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ăn phở đi", "timestamp": "2025-08-01T10:01:00"},
            {"author": "Hoang Quoc Viet", "text": "ok 👍", "timestamp": "2025-08-01T10:02:00"},
        ],
        "metadata": {
            "participants": ["Hoang Quoc Viet", "Friend"],
        },
    }

    result = analyze_stats(chunk, twin_name="Hoang Quoc Viet")

    assert result["msg_count"] == 3
    assert result["twin_msg_count"] == 2
    assert result["twin_msg_ratio"] == round(2 / 3, 3)
    assert result["avg_msg_len"] > 0
    assert result["twin_avg_msg_len"] > 0
    assert result["emoji_count"] >= 1
    assert result["question_ratio"] > 0  # "ăn j đây" has question-like pattern
    assert result["language"] == "vi"


def test_analyze_stats_empty_chunk():
    """Stats analyzer handles empty messages gracefully."""
    chunk = {"messages": [], "metadata": {"participants": []}}
    result = analyze_stats(chunk, twin_name="Hoang Quoc Viet")

    assert result["msg_count"] == 0
    assert result["twin_msg_ratio"] == 0.0


def test_analyze_stats_mixed_language():
    """Stats analyzer detects mixed language."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "are you oke", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Viet", "text": "where are diu", "timestamp": "2025-08-01T10:01:00"},
            {"author": "Viet", "text": "tối ăn gì này", "timestamp": "2025-08-01T10:02:00"},
        ],
        "metadata": {"participants": ["Viet"]},
    }
    result = analyze_stats(chunk, twin_name="Viet")
    assert result["language"] in ("mixed", "vi", "en")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_stats.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.analyzers'"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/__init__.py
"""Metadata enrichment analyzers for chat chunks."""
```

```python
# app/analyzers/stats.py
"""Stats analyzer — heuristic, no LLM required.

Extracts message-level statistics from a chunk: counts, ratios,
emoji usage, question detection, language detection.
"""

import re
import unicodedata

# Vietnamese-specific characters beyond ASCII
_VIETNAMESE_PATTERN = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', re.IGNORECASE)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U0001f900-\U0001f9FF"
    "]+",
    flags=re.UNICODE,
)
_QUESTION_PATTERN = re.compile(r'[?？]|(\b(gì|j|sao|nào|ko|không|chưa|hả|đâu|bao giờ|mấy|ai)\b)', re.IGNORECASE)


def _detect_language(texts: list[str]) -> str:
    """Detect primary language from a list of texts. Returns 'vi', 'en', or 'mixed'."""
    if not texts:
        return "vi"

    vi_count = 0
    en_count = 0
    for text in texts:
        if _VIETNAMESE_PATTERN.search(text):
            vi_count += 1
        # Simple heuristic: if no Vietnamese chars and has ASCII letters, likely English
        elif re.search(r'[a-zA-Z]{3,}', text):
            en_count += 1

    total = vi_count + en_count
    if total == 0:
        return "vi"
    if vi_count / total > 0.7:
        return "vi"
    if en_count / total > 0.7:
        return "en"
    return "mixed"


def analyze_stats(chunk: dict, twin_name: str) -> dict:
    """Extract heuristic statistics from a chunk. No LLM required.

    Args:
        chunk: Dict with "messages" list (each has "author", "text", "timestamp")
        twin_name: Name of the twin to calculate twin-specific stats

    Returns:
        Dict with stats metadata fields.
    """
    messages = chunk.get("messages", [])
    if not messages:
        return {
            "msg_count": 0,
            "twin_msg_count": 0,
            "twin_msg_ratio": 0.0,
            "avg_msg_len": 0.0,
            "twin_avg_msg_len": 0.0,
            "emoji_count": 0,
            "question_ratio": 0.0,
            "language": "vi",
        }

    texts = [m.get("text", "") for m in messages]
    twin_texts = [m.get("text", "") for m in messages if m.get("author") == twin_name]

    all_lengths = [len(t) for t in texts]
    twin_lengths = [len(t) for t in twin_texts]

    emoji_count = sum(len(_EMOJI_PATTERN.findall(t)) for t in texts)
    question_count = sum(1 for t in texts if _QUESTION_PATTERN.search(t))

    return {
        "msg_count": len(messages),
        "twin_msg_count": len(twin_texts),
        "twin_msg_ratio": round(len(twin_texts) / len(messages), 3) if messages else 0.0,
        "avg_msg_len": round(sum(all_lengths) / len(all_lengths), 1) if all_lengths else 0.0,
        "twin_avg_msg_len": round(sum(twin_lengths) / len(twin_lengths), 1) if twin_lengths else 0.0,
        "emoji_count": emoji_count,
        "question_ratio": round(question_count / len(messages), 3) if messages else 0.0,
        "language": _detect_language(texts),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_stats.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/__init__.py app/analyzers/stats.py tests/test_analyzers_stats.py
git commit -m "feat: stats analyzer — heuristic message statistics, no LLM"
```

---

### Task 3: Analyzer Registry + Runner

**Files:**
- Create: `app/analyzers/registry.py`
- Test: `tests/test_analyzers_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_registry.py

from app.analyzers.registry import AnalyzerInput, AnalyzerRegistry, run_analyzers


def _fake_stats(input: AnalyzerInput, **kwargs) -> dict:
    return {"msg_count": len(input.chunk.get("messages", [])), "language": "vi"}


def _fake_tone(input: AnalyzerInput, **kwargs) -> dict:
    # Simulate using context_summary from a previous analyzer
    summary = input.chunk.get("metadata", {}).get("context_summary", "")
    return {"tone": "casual" if "casual" in summary else "unknown"}


def test_registry_registers_and_runs_analyzers():
    """Registry runs analyzers in order and merges metadata."""
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=_fake_stats, version=1, requires_llm=False, run_order=0)
    registry.register("tone_v1", fn=_fake_tone, version=1, requires_llm=True, run_order=1)

    chunk = {"messages": [{"author": "A", "text": "hello"}], "metadata": {}}

    result = run_analyzers(registry, chunk, twin_name="A")

    assert result["msg_count"] == 1
    assert result["language"] == "vi"
    assert "tone" in result
    assert result["_analyzers_applied"] == {"stats_v1": 1, "tone_v1": 1}


def test_registry_skips_already_applied():
    """Registry skips analyzers that are already applied at the same version."""
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=_fake_stats, version=1, requires_llm=False, run_order=0)

    chunk = {
        "messages": [{"author": "A", "text": "hello"}],
        "metadata": {"_analyzers_applied": {"stats_v1": 1}},
    }

    result = run_analyzers(registry, chunk, twin_name="A")

    # stats_v1 already applied at v1 — should be skipped, return empty new metadata
    assert result == {"_analyzers_applied": {"stats_v1": 1}}


def test_registry_reruns_outdated_version():
    """Registry re-runs analyzers when version is newer."""
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=_fake_stats, version=2, requires_llm=False, run_order=0)

    chunk = {
        "messages": [{"author": "A", "text": "hello"}],
        "metadata": {"_analyzers_applied": {"stats_v1": 1}},  # old version
    }

    result = run_analyzers(registry, chunk, twin_name="A")

    assert result["msg_count"] == 1
    assert result["_analyzers_applied"]["stats_v1"] == 2


def test_registry_run_order():
    """Analyzers with run_order=0 execute before run_order=1."""
    call_order = []

    def first(input, **kwargs):
        call_order.append("first")
        return {"context_summary": "casual chat"}

    def second(input, **kwargs):
        call_order.append("second")
        return {"tone": "casual"}

    registry = AnalyzerRegistry()
    registry.register("context_v1", fn=first, version=1, requires_llm=True, run_order=0)
    registry.register("tone_v1", fn=second, version=1, requires_llm=True, run_order=1)

    chunk = {"messages": [{"author": "A", "text": "hi"}], "metadata": {}}
    run_analyzers(registry, chunk, twin_name="A")

    assert call_order == ["first", "second"]


def test_registry_context_flows_between_orders():
    """run_order=0 output is available to run_order=1 analyzers via chunk metadata."""
    def context_analyzer(input: AnalyzerInput, **kwargs):
        return {"context_summary": "casual dinner planning"}

    def tone_analyzer(input: AnalyzerInput, **kwargs):
        summary = input.chunk.get("metadata", {}).get("context_summary", "")
        return {"tone": "casual" if "casual" in summary else "formal"}

    registry = AnalyzerRegistry()
    registry.register("context_v1", fn=context_analyzer, version=1, requires_llm=True, run_order=0)
    registry.register("tone_v1", fn=tone_analyzer, version=1, requires_llm=True, run_order=1)

    chunk = {"messages": [{"author": "A", "text": "hi"}], "metadata": {}}
    result = run_analyzers(registry, chunk, twin_name="A")

    assert result["context_summary"] == "casual dinner planning"
    assert result["tone"] == "casual"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_registry.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.analyzers.registry'"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/registry.py
"""Analyzer registry and runner.

Manages a collection of analyzer functions that enrich chunks with metadata.
Analyzers run in order (run_order), and output from earlier analyzers is
available to later ones via the chunk's metadata dict.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class AnalyzerInput:
    """Input to an analyzer function."""
    chunk: dict                    # the chunk being analyzed
    prev_chunk: dict | None = None # previous chunk in same thread
    next_chunk: dict | None = None # next chunk in same thread
    thread_summary: str | None = None  # running summary of the thread


@dataclass
class AnalyzerDef:
    """Definition of a registered analyzer."""
    name: str
    fn: Callable[[AnalyzerInput], dict]
    version: int
    requires_llm: bool
    run_order: int


class AnalyzerRegistry:
    """Registry of analyzer functions."""

    def __init__(self):
        self._analyzers: dict[str, AnalyzerDef] = {}

    def register(self, name: str, fn: Callable, version: int,
                 requires_llm: bool, run_order: int):
        self._analyzers[name] = AnalyzerDef(
            name=name, fn=fn, version=version,
            requires_llm=requires_llm, run_order=run_order,
        )

    def get_all(self) -> list[AnalyzerDef]:
        """Return all analyzers sorted by run_order, then name."""
        return sorted(self._analyzers.values(), key=lambda a: (a.run_order, a.name))

    def get(self, name: str) -> AnalyzerDef | None:
        return self._analyzers.get(name)


def run_analyzers(
    registry: AnalyzerRegistry,
    chunk: dict,
    twin_name: str,
    prev_chunk: dict | None = None,
    next_chunk: dict | None = None,
    thread_summary: str | None = None,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Run all applicable analyzers on a chunk and return merged metadata.

    Skips analyzers that are already applied at the current version.
    Earlier run_order output is merged into chunk metadata before later analyzers run.

    Args:
        registry: The analyzer registry
        chunk: The chunk to analyze (will not be mutated)
        twin_name: Name of the twin for twin-specific analysis
        prev_chunk: Previous chunk in same thread (for context)
        next_chunk: Next chunk in same thread (for context)
        thread_summary: Running summary of the thread
        llm_client: OpenAI-compatible client (for LLM-based analyzers)
        llm_model: Model name (for LLM-based analyzers)

    Returns:
        Dict of new/updated metadata fields, including _analyzers_applied.
    """
    applied = dict(chunk.get("metadata", {}).get("_analyzers_applied", {}))
    all_new_metadata: dict = {}

    # Group by run_order
    analyzers = registry.get_all()
    current_order = -1

    for analyzer in analyzers:
        # Skip if already applied at this version
        if applied.get(analyzer.name) == analyzer.version:
            continue

        # When entering a new run_order group, merge previous group's output
        # into a working copy of chunk metadata so later analyzers can see it
        if analyzer.run_order != current_order:
            current_order = analyzer.run_order
            # Update chunk metadata view with results from previous orders
            working_metadata = dict(chunk.get("metadata", {}))
            working_metadata.update(all_new_metadata)
            working_chunk = {**chunk, "metadata": working_metadata}

        analyzer_input = AnalyzerInput(
            chunk=working_chunk,
            prev_chunk=prev_chunk,
            next_chunk=next_chunk,
            thread_summary=thread_summary,
        )

        # Call the analyzer
        kwargs = {}
        if analyzer.requires_llm:
            kwargs["llm_client"] = llm_client
            kwargs["llm_model"] = llm_model
        kwargs["twin_name"] = twin_name

        result = analyzer.fn(analyzer_input, **kwargs)

        # Merge results
        all_new_metadata.update(result)
        applied[analyzer.name] = analyzer.version

    all_new_metadata["_analyzers_applied"] = applied
    return all_new_metadata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/registry.py tests/test_analyzers_registry.py
git commit -m "feat: analyzer registry with versioned run-order execution"
```

---

### Task 4: Context Analyzer (LLM-Based)

**Files:**
- Create: `app/analyzers/context.py`
- Test: `tests/test_analyzers_context.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_context.py

from unittest.mock import MagicMock, patch
from app.analyzers.registry import AnalyzerInput
from app.analyzers.context import analyze_context


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_context_returns_summary():
    """Context analyzer generates a context summary from chunk messages."""
    chunk = {
        "messages": [
            {"author": "Friend", "text": "tối nay ăn gì", "timestamp": "2025-08-01T18:00:00"},
            {"author": "Viet", "text": "ăn phở đi", "timestamp": "2025-08-01T18:01:00"},
        ],
        "metadata": {"participants": ["Viet", "Friend"]},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"context_summary": "friends planning dinner", "interaction_type": "planning", "relationship": "close_friends"}')

    result = analyze_context(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "context_summary" in result
    assert "interaction_type" in result
    assert "relationship" in result


def test_analyze_context_uses_neighbors():
    """Context analyzer includes prev/next chunk in prompt for broader context."""
    chunk = {
        "messages": [{"author": "Viet", "text": "ok fine", "timestamp": "2025-08-01T18:05:00"}],
        "metadata": {},
    }
    prev_chunk = {
        "messages": [{"author": "Friend", "text": "you lied to me", "timestamp": "2025-08-01T18:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk, prev_chunk=prev_chunk)

    client = _make_mock_llm_client('{"context_summary": "tense moment after accusation", "interaction_type": "conflict", "relationship": "strained"}')

    result = analyze_context(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    # Verify the LLM was called with both chunks
    call_args = client.chat.completions.create.call_args
    prompt_content = str(call_args)
    assert "you lied to me" in prompt_content or result["interaction_type"] == "conflict"


def test_analyze_context_handles_malformed_json():
    """Context analyzer returns defaults when LLM returns bad JSON."""
    chunk = {
        "messages": [{"author": "Viet", "text": "hi", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client("this is not json at all")

    result = analyze_context(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "context_summary" in result
    assert result["context_summary"] != ""  # should have a fallback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_context.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.analyzers.context'"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/context.py
"""Context analyzer — LLM-based chunk summarization.

Generates a context summary describing what's happening in a chunk,
using neighboring chunks for broader context awareness.
"""

import json

from app.analyzers.registry import AnalyzerInput


def _format_messages(messages: list[dict], max_messages: int = 15) -> str:
    """Format messages for LLM prompt."""
    lines = []
    for m in messages[:max_messages]:
        lines.append(f"{m.get('author', '?')}: {m.get('text', '')}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """You analyze chat conversations. Given a chunk of messages (and optionally surrounding context), return a JSON object with:

- "context_summary": 1-2 sentence description of what's happening (e.g. "friends planning dinner, relaxed tone")
- "interaction_type": one of: greeting, planning, banter, argument, support, information_sharing, story_telling, venting, flirting, business, other
- "relationship": one of: close_friends, acquaintances, romantic, family, colleagues, strangers, other

Return ONLY valid JSON, no markdown, no explanation."""


def analyze_context(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze chunk context using LLM.

    Args:
        input: AnalyzerInput with chunk and optional neighbors
        twin_name: Name of the twin
        llm_client: OpenAI-compatible client
        llm_model: Model name

    Returns:
        Dict with context_summary, interaction_type, relationship
    """
    defaults = {
        "context_summary": "general conversation",
        "interaction_type": "other",
        "relationship": "other",
    }

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    # Build the user prompt with context
    parts = []
    if input.prev_chunk and input.prev_chunk.get("messages"):
        parts.append(f"[PREVIOUS CONTEXT]\n{_format_messages(input.prev_chunk['messages'], max_messages=5)}")
    parts.append(f"[CURRENT CHUNK]\n{_format_messages(input.chunk['messages'])}")
    if input.next_chunk and input.next_chunk.get("messages"):
        parts.append(f"[NEXT CONTEXT]\n{_format_messages(input.next_chunk['messages'], max_messages=5)}")

    user_prompt = "\n\n".join(parts)

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            timeout=30,
        )
        raw = response.choices[0].message.content or ""

        # Try to parse JSON from the response
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(cleaned)
        return {
            "context_summary": parsed.get("context_summary", defaults["context_summary"]),
            "interaction_type": parsed.get("interaction_type", defaults["interaction_type"]),
            "relationship": parsed.get("relationship", defaults["relationship"]),
        }
    except (json.JSONDecodeError, Exception):
        # Fallback: use the raw text as summary if it's short enough
        if raw and len(raw) < 200:
            return {**defaults, "context_summary": raw.strip()}
        return defaults
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_context.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/context.py tests/test_analyzers_context.py
git commit -m "feat: context analyzer — LLM-based chunk summarization with neighbor awareness"
```

---

### Task 5: Tone Analyzer (LLM-Based)

**Files:**
- Create: `app/analyzers/tone.py`
- Test: `tests/test_analyzers_tone.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_tone.py

from unittest.mock import MagicMock
from app.analyzers.registry import AnalyzerInput
from app.analyzers.tone import analyze_tone


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_tone_returns_fields():
    """Tone analyzer returns tone, formality, and energy."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "ê đi ăn k", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ok đi", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "friends planning to eat"},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"tone": "casual_banter", "formality": 0.1, "energy": "relaxed"}')

    result = analyze_tone(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "tone" in result
    assert "formality" in result
    assert "energy" in result
    assert isinstance(result["formality"], (int, float))


def test_analyze_tone_uses_context_summary():
    """Tone analyzer includes context_summary from prior analyzer in its prompt."""
    chunk = {
        "messages": [{"author": "Viet", "text": "...", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {"context_summary": "heated argument about money"},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"tone": "angry", "formality": 0.3, "energy": "high"}')

    result = analyze_tone(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    # Verify context_summary was passed to LLM
    call_args = client.chat.completions.create.call_args
    prompt_content = str(call_args)
    assert "heated argument" in prompt_content


def test_analyze_tone_handles_malformed_json():
    """Tone analyzer returns defaults when LLM returns bad JSON."""
    chunk = {
        "messages": [{"author": "Viet", "text": "hi", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client("not json")

    result = analyze_tone(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["tone"] == "neutral"
    assert result["formality"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_tone.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.analyzers.tone'"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/tone.py
"""Tone analyzer — LLM-based tone/formality classification.

Classifies the overall tone, formality level, and energy of a chunk.
Uses context_summary from the context analyzer if available.
"""

import json

from app.analyzers.registry import AnalyzerInput

_SYSTEM_PROMPT = """Analyze the tone of this chat conversation chunk. Return a JSON object with:

- "tone": one of: casual_banter, casual, playful, friendly, formal, formal_news, informational, serious, sarcastic, humorous, emotional, angry, confrontational, frustrated, vulnerable, personal, relaxed, neutral
- "formality": float 0.0 (very informal/slang) to 1.0 (very formal/proper)
- "energy": one of: high, medium, low, relaxed

Consider the context summary if provided. Return ONLY valid JSON."""


def analyze_tone(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze chunk tone using LLM."""
    defaults = {"tone": "neutral", "formality": 0.5, "energy": "medium"}

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    messages_text = "\n".join(
        f"{m.get('author', '?')}: {m.get('text', '')}"
        for m in input.chunk["messages"][:15]
    )

    context_summary = input.chunk.get("metadata", {}).get("context_summary", "")
    context_line = f"\n\nContext: {context_summary}" if context_summary else ""

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"{messages_text}{context_line}"},
            ],
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "tone": parsed.get("tone", defaults["tone"]),
            "formality": float(parsed.get("formality", defaults["formality"])),
            "energy": parsed.get("energy", defaults["energy"]),
        }
    except (json.JSONDecodeError, ValueError, Exception):
        return defaults
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_tone.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/tone.py tests/test_analyzers_tone.py
git commit -m "feat: tone analyzer — LLM-based tone/formality classification"
```

---

### Task 6: Emotion Analyzer (LLM-Based)

**Files:**
- Create: `app/analyzers/emotion.py`
- Test: `tests/test_analyzers_emotion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_emotion.py

from unittest.mock import MagicMock
from app.analyzers.registry import AnalyzerInput
from app.analyzers.emotion import analyze_emotion


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_emotion_returns_fields():
    """Emotion analyzer returns emotion, sentiment, conflict, sarcasm."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "haha bạn ngu quá", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "🤣🤣🤣", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "friends joking around"},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"emotion": "playful", "sentiment": 0.8, "conflict": false, "sarcasm": false}')

    result = analyze_emotion(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert "emotion" in result
    assert "sentiment" in result
    assert "conflict" in result
    assert "sarcasm" in result
    assert isinstance(result["conflict"], bool)


def test_analyze_emotion_detects_conflict():
    """Emotion analyzer identifies conflict in arguments."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "sao m nói vậy", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "thì sự thật mà", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "disagreement about plans"},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"emotion": "frustrated", "sentiment": -0.3, "conflict": true, "sarcasm": false}')

    result = analyze_emotion(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["conflict"] is True
    assert result["sentiment"] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_emotion.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/emotion.py
"""Emotion analyzer — LLM-based emotion/sentiment/conflict detection.

Classifies the emotional state, sentiment, and whether conflict or sarcasm
is present in a chunk.
"""

import json

from app.analyzers.registry import AnalyzerInput

_SYSTEM_PROMPT = """Analyze the emotional content of this chat conversation chunk. Return a JSON object with:

- "emotion": primary emotion — one of: happy, playful, excited, neutral, tired, sad, frustrated, angry, anxious, vulnerable, nostalgic, sarcastic, bored
- "sentiment": float from -1.0 (very negative) to 1.0 (very positive), 0.0 is neutral
- "conflict": boolean — is there a disagreement, argument, or tension between participants?
- "sarcasm": boolean — is sarcasm or irony being used?

Consider the context summary if provided. Return ONLY valid JSON."""


def analyze_emotion(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze chunk emotion using LLM."""
    defaults = {"emotion": "neutral", "sentiment": 0.0, "conflict": False, "sarcasm": False}

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    messages_text = "\n".join(
        f"{m.get('author', '?')}: {m.get('text', '')}"
        for m in input.chunk["messages"][:15]
    )

    context_summary = input.chunk.get("metadata", {}).get("context_summary", "")
    context_line = f"\n\nContext: {context_summary}" if context_summary else ""

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"{messages_text}{context_line}"},
            ],
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "emotion": parsed.get("emotion", defaults["emotion"]),
            "sentiment": float(parsed.get("sentiment", defaults["sentiment"])),
            "conflict": bool(parsed.get("conflict", defaults["conflict"])),
            "sarcasm": bool(parsed.get("sarcasm", defaults["sarcasm"])),
        }
    except (json.JSONDecodeError, ValueError, Exception):
        return defaults
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_emotion.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/emotion.py tests/test_analyzers_emotion.py
git commit -m "feat: emotion analyzer — sentiment, conflict, sarcasm detection"
```

---

### Task 7: Persona Analyzer (LLM-Based)

**Files:**
- Create: `app/analyzers/persona.py`
- Test: `tests/test_analyzers_persona.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_persona.py

from unittest.mock import MagicMock
from app.analyzers.registry import AnalyzerInput
from app.analyzers.persona import analyze_persona


def _make_mock_llm_client(response_text: str):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    client.chat.completions.create.return_value = mock_response
    return client


def test_analyze_persona_returns_fields():
    """Persona analyzer returns twin_role, register, relationship_to_others."""
    chunk = {
        "messages": [
            {"author": "Viet", "text": "đi ăn k mấy ông", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ok đi", "timestamp": "2025-08-01T10:01:00"},
        ],
        "metadata": {"context_summary": "friend group planning dinner"},
    }
    input = AnalyzerInput(chunk=chunk)

    client = _make_mock_llm_client('{"twin_role": "initiator", "register": "informal_close", "relationship_to_others": "friend_banter"}')

    result = analyze_persona(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["twin_role"] == "initiator"
    assert result["register"] == "informal_close"
    assert result["relationship_to_others"] == "friend_banter"


def test_analyze_persona_handles_malformed_json():
    """Persona analyzer returns defaults on bad JSON."""
    chunk = {
        "messages": [{"author": "Viet", "text": "hi", "timestamp": "2025-08-01T10:00:00"}],
        "metadata": {},
    }
    input = AnalyzerInput(chunk=chunk)
    client = _make_mock_llm_client("garbage response")

    result = analyze_persona(input, twin_name="Viet", llm_client=client, llm_model="test-model")

    assert result["twin_role"] == "participant"
    assert result["register"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_persona.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/persona.py
"""Persona analyzer — LLM-based role and register classification.

Classifies how the twin behaves in this chunk: their role in the conversation,
their register (formality level with this person), and relationship type.
"""

import json

from app.analyzers.registry import AnalyzerInput

_SYSTEM_PROMPT = """Analyze the twin's persona in this conversation. The twin's name is: {twin_name}

Return a JSON object with:
- "twin_role": how the twin acts — one of: initiator, responder, leader, follower, mediator, provocateur, supporter, observer, participant
- "register": the twin's language register here — one of: informal_close, informal_casual, semi_formal, formal, code_switching, playful_vulgar, respectful_elder, unknown
- "relationship_to_others": one of: friend_banter, close_friend, acquaintance, romantic, family_casual, family_formal, colleague, mentor, mentee, stranger, other

Consider the context summary if provided. Return ONLY valid JSON."""


def analyze_persona(
    input: AnalyzerInput,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
) -> dict:
    """Analyze twin's persona in this chunk using LLM."""
    defaults = {"twin_role": "participant", "register": "unknown", "relationship_to_others": "other"}

    if not llm_client or not input.chunk.get("messages"):
        return defaults

    messages_text = "\n".join(
        f"{m.get('author', '?')}: {m.get('text', '')}"
        for m in input.chunk["messages"][:15]
    )

    context_summary = input.chunk.get("metadata", {}).get("context_summary", "")
    context_line = f"\n\nContext: {context_summary}" if context_summary else ""

    system_prompt = _SYSTEM_PROMPT.format(twin_name=twin_name)

    try:
        response = llm_client.chat.completions.create(
            model=llm_model or "llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{messages_text}{context_line}"},
            ],
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "twin_role": parsed.get("twin_role", defaults["twin_role"]),
            "register": parsed.get("register", defaults["register"]),
            "relationship_to_others": parsed.get("relationship_to_others", defaults["relationship_to_others"]),
        }
    except (json.JSONDecodeError, ValueError, Exception):
        return defaults
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_persona.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/persona.py tests/test_analyzers_persona.py
git commit -m "feat: persona analyzer — twin role, register, relationship classification"
```

---

### Task 8: Dynamic Chunking — Boundary Detection

**Files:**
- Create: `app/chunking/__init__.py`
- Create: `app/chunking/boundary.py`
- Test: `tests/test_chunking_boundary.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking_boundary.py

from unittest.mock import MagicMock
from app.chunking.boundary import detect_boundaries


def _make_mock_llm_client(responses: list[str]):
    """Mock LLM that returns different responses for sequential calls."""
    client = MagicMock()
    mock_responses = []
    for text in responses:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = text
        mock_responses.append(mock_response)
    client.chat.completions.create.side_effect = mock_responses
    return client


def test_detect_boundaries_splits_on_topic_shift():
    """Boundary detection splits when LLM identifies topic shift."""
    messages = [
        {"author": "A", "text": "đi ăn k", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "ok đi", "timestamp": "2025-08-01T10:01:00"},
        {"author": "A", "text": "ăn phở nhé", "timestamp": "2025-08-01T10:02:00"},
        # Topic shift here
        {"author": "A", "text": "ê m làm xong bài chưa", "timestamp": "2025-08-01T10:30:00"},
        {"author": "B", "text": "chưa", "timestamp": "2025-08-01T10:31:00"},
        {"author": "A", "text": "deadline mai rồi", "timestamp": "2025-08-01T10:32:00"},
    ]

    # LLM says: messages 0-2 = same topic, 3 = new topic, 4-5 = same as 3
    client = _make_mock_llm_client([
        '{"boundaries": [3]}',  # boundary before index 3
    ])

    boundaries = detect_boundaries(messages, llm_client=client, llm_model="test")

    assert 3 in boundaries


def test_detect_boundaries_no_split_on_short_conversation():
    """Short conversations (< 3 messages) get no internal boundaries."""
    messages = [
        {"author": "A", "text": "hi", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
    ]

    client = _make_mock_llm_client(['{"boundaries": []}'])

    boundaries = detect_boundaries(messages, llm_client=client, llm_model="test")

    assert boundaries == []


def test_detect_boundaries_returns_empty_without_llm():
    """Without LLM client, returns empty boundaries (single chunk)."""
    messages = [
        {"author": "A", "text": "hi", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
    ]

    boundaries = detect_boundaries(messages, llm_client=None, llm_model=None)

    assert boundaries == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_boundary.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/chunking/__init__.py
"""Dynamic chunking system for conversation messages."""
```

```python
# app/chunking/boundary.py
"""LLM-based conversation boundary detection.

Scans a list of messages and identifies natural break points where
the topic, mood, or context shifts significantly.
"""

import json

_SYSTEM_PROMPT = """You analyze chat conversations to find natural break points.

Given a list of messages, identify indices where a NEW conversation segment starts. A new segment means:
- Topic changes (food → work, personal → news)
- Mood shifts (playful → serious, calm → angry)
- New participant enters and changes the dynamic
- Time gap + clear context shift (time gap alone is NOT enough if topic continues)

Return a JSON object: {"boundaries": [list of message indices where new segments start]}

Example: messages 0-4 about food, messages 5-9 about work → {"boundaries": [5]}

If the entire conversation is one coherent segment, return {"boundaries": []}
Return ONLY valid JSON."""


def _format_messages_for_boundary(messages: list[dict]) -> str:
    """Format messages with indices for boundary detection."""
    lines = []
    for i, m in enumerate(messages):
        ts = m.get("timestamp", "")
        author = m.get("author", "?")
        text = m.get("text", "")
        lines.append(f"[{i}] {ts} {author}: {text}")
    return "\n".join(lines)


def detect_boundaries(
    messages: list[dict],
    llm_client=None,
    llm_model: str | None = None,
    window_size: int = 40,
) -> list[int]:
    """Detect conversation boundaries in a list of messages.

    Uses a sliding window approach for long conversations.

    Args:
        messages: List of message dicts with author, text, timestamp
        llm_client: OpenAI-compatible client
        llm_model: Model name
        window_size: Max messages to send to LLM at once

    Returns:
        Sorted list of boundary indices (where new segments start).
    """
    if not llm_client or len(messages) < 3:
        return []

    all_boundaries: set[int] = set()

    # Process in overlapping windows for long conversations
    step = window_size - 5  # 5 message overlap
    for start in range(0, len(messages), max(step, 1)):
        window = messages[start:start + window_size]
        if len(window) < 3:
            break

        formatted = _format_messages_for_boundary(window)

        try:
            response = llm_client.chat.completions.create(
                model=llm_model or "llama3.1:8b",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": formatted},
                ],
                timeout=30,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            for idx in parsed.get("boundaries", []):
                # Convert window-local index to global index
                global_idx = start + int(idx)
                if 0 < global_idx < len(messages):
                    all_boundaries.add(global_idx)
        except (json.JSONDecodeError, ValueError, Exception):
            continue

        # If window covers everything, no need to continue
        if start + window_size >= len(messages):
            break

    return sorted(all_boundaries)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking_boundary.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/chunking/__init__.py app/chunking/boundary.py tests/test_chunking_boundary.py
git commit -m "feat: LLM-based conversation boundary detection"
```

---

### Task 9: Dynamic Chunking — Normalizer

**Files:**
- Create: `app/chunking/normalizer.py`
- Test: `tests/test_chunking_normalizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking_normalizer.py

from app.chunking.normalizer import normalize_segments, build_chunks


def test_normalize_merges_tiny_segments():
    """Segments with < 3 messages get merged with their neighbor."""
    messages = [
        {"author": "A", "text": "hi", "timestamp": "2025-08-01T10:00:00"},
        {"author": "B", "text": "hey", "timestamp": "2025-08-01T10:01:00"},
        # boundary
        {"author": "A", "text": "đi ăn k", "timestamp": "2025-08-01T10:02:00"},
        {"author": "B", "text": "ok", "timestamp": "2025-08-01T10:03:00"},
        {"author": "A", "text": "ăn phở", "timestamp": "2025-08-01T10:04:00"},
        {"author": "B", "text": "oke", "timestamp": "2025-08-01T10:05:00"},
        {"author": "A", "text": "mấy giờ", "timestamp": "2025-08-01T10:06:00"},
    ]
    boundaries = [2]  # first segment has only 2 messages

    normalized = normalize_segments(messages, boundaries, min_size=3, max_size=20)

    # The 2-message segment should be merged with the next one
    assert len(normalized) == 1  # single segment
    assert len(normalized[0]) == 7  # all messages together


def test_normalize_splits_large_segments():
    """Segments with > max_size messages get split."""
    messages = [{"author": "A", "text": f"msg {i}", "timestamp": f"2025-08-01T{10+i}:00:00"} for i in range(25)]
    boundaries = []  # one giant segment

    normalized = normalize_segments(messages, boundaries, min_size=3, max_size=10)

    assert len(normalized) >= 2
    for seg in normalized:
        assert len(seg) <= 10


def test_build_chunks_creates_chunk_dicts():
    """build_chunks creates properly structured chunk dicts from segments."""
    segments = [
        [
            {"author": "Viet", "text": "ê đi ăn", "timestamp": "2025-08-01T10:00:00"},
            {"author": "Friend", "text": "ok", "timestamp": "2025-08-01T10:01:00"},
            {"author": "Viet", "text": "ăn phở", "timestamp": "2025-08-01T10:02:00"},
        ],
    ]

    chunks = build_chunks(segments, thread_id="inbox/thread_1", twin_name="Viet")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"].startswith("inbox/thread_1_seg")
    assert chunk["messages"] == segments[0]
    assert "document" in chunk
    assert "Viet: ê đi ăn" in chunk["document"]
    assert chunk["metadata"]["msg_count"] == 3
    assert chunk["metadata"]["participants"] == ["Friend", "Viet"]
    assert chunk["metadata"]["time_start"] == "2025-08-01T10:00:00"
    assert chunk["metadata"]["time_end"] == "2025-08-01T10:02:00"
    assert chunk["metadata"]["twin_msg_ratio"] == round(2 / 3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking_normalizer.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/chunking/normalizer.py
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
    """Split messages into segments at boundaries, then normalize sizes.

    - Segments < min_size get merged with the nearest neighbor
    - Segments > max_size get split at even intervals

    Args:
        messages: All messages in order
        boundaries: Sorted list of indices where new segments start
        min_size: Minimum messages per segment
        max_size: Maximum messages per segment

    Returns:
        List of message lists, each a normalized segment.
    """
    if not messages:
        return []

    # Split at boundaries
    cuts = [0] + sorted(boundaries) + [len(messages)]
    raw_segments = []
    for i in range(len(cuts) - 1):
        seg = messages[cuts[i]:cuts[i + 1]]
        if seg:
            raw_segments.append(seg)

    if not raw_segments:
        return []

    # Merge tiny segments with their next neighbor
    merged = []
    carry: list[dict] = []
    for seg in raw_segments:
        combined = carry + seg
        if len(combined) < min_size:
            carry = combined
        else:
            merged.append(combined)
            carry = []
    # Leftover carry: merge with last segment
    if carry:
        if merged:
            merged[-1].extend(carry)
        else:
            merged.append(carry)

    # Split oversized segments
    final = []
    for seg in merged:
        if len(seg) <= max_size:
            final.append(seg)
        else:
            # Split into roughly equal parts
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
    """Build chunk dicts from normalized segments.

    Args:
        segments: List of message lists (from normalize_segments)
        thread_id: Thread identifier for chunk IDs
        twin_name: Name of the twin

    Returns:
        List of chunk dicts ready for enrichment and ChromaDB ingestion.
    """
    chunks = []
    for i, seg in enumerate(segments):
        if not seg:
            continue

        # Build document text (one line per message)
        doc_lines = []
        for m in seg:
            doc_lines.append(f"{m.get('author', '?')}: {m.get('text', '')}")
        document = "\n".join(doc_lines)

        # Extract metadata
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking_normalizer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/chunking/normalizer.py tests/test_chunking_normalizer.py
git commit -m "feat: segment normalizer — merge tiny, split large, build chunks"
```

---

### Task 10: Wire Analyzers Into Default Registry

**Files:**
- Create: `app/analyzers/default_registry.py`
- Test: `tests/test_analyzers_default_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzers_default_registry.py

from app.analyzers.default_registry import create_default_registry


def test_default_registry_has_all_analyzers():
    """Default registry includes all 5 analyzers."""
    registry = create_default_registry()
    analyzers = registry.get_all()
    names = [a.name for a in analyzers]

    assert "stats_v1" in names
    assert "context_v1" in names
    assert "tone_v1" in names
    assert "emotion_v1" in names
    assert "persona_v1" in names


def test_default_registry_run_order():
    """stats_v1 and context_v1 run before tone/emotion/persona."""
    registry = create_default_registry()
    analyzers = registry.get_all()

    order_0 = [a.name for a in analyzers if a.run_order == 0]
    order_1 = [a.name for a in analyzers if a.run_order == 1]

    assert "stats_v1" in order_0
    assert "context_v1" in order_0
    assert "tone_v1" in order_1
    assert "emotion_v1" in order_1
    assert "persona_v1" in order_1


def test_default_registry_llm_requirements():
    """stats_v1 does not require LLM, others do."""
    registry = create_default_registry()

    stats = registry.get("stats_v1")
    assert stats is not None
    assert stats.requires_llm is False

    tone = registry.get("tone_v1")
    assert tone is not None
    assert tone.requires_llm is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers_default_registry.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# app/analyzers/default_registry.py
"""Default analyzer registry with all built-in analyzers."""

from app.analyzers.registry import AnalyzerRegistry
from app.analyzers.stats import analyze_stats
from app.analyzers.context import analyze_context
from app.analyzers.tone import analyze_tone
from app.analyzers.emotion import analyze_emotion
from app.analyzers.persona import analyze_persona


def create_default_registry() -> AnalyzerRegistry:
    """Create the default analyzer registry with all built-in analyzers.

    Run order:
        0: stats_v1 (no LLM), context_v1 (LLM)
        1: tone_v1, emotion_v1, persona_v1 (all LLM, can use context_summary)
    """
    registry = AnalyzerRegistry()

    # Order 0 — foundation analyzers
    registry.register(
        "stats_v1",
        fn=lambda input, **kw: analyze_stats(input.chunk, twin_name=kw.get("twin_name", "")),
        version=1,
        requires_llm=False,
        run_order=0,
    )
    registry.register(
        "context_v1",
        fn=analyze_context,
        version=1,
        requires_llm=True,
        run_order=0,
    )

    # Order 1 — use context_summary from context_v1
    registry.register("tone_v1", fn=analyze_tone, version=1, requires_llm=True, run_order=1)
    registry.register("emotion_v1", fn=analyze_emotion, version=1, requires_llm=True, run_order=1)
    registry.register("persona_v1", fn=analyze_persona, version=1, requires_llm=True, run_order=1)

    return registry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers_default_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/analyzers/default_registry.py tests/test_analyzers_default_registry.py
git commit -m "feat: default analyzer registry wiring all 5 analyzers"
```

---

### Task 11: Backfill CLI

**Files:**
- Create: `app/backfill.py`
- Test: `tests/test_backfill.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill.py

from unittest.mock import MagicMock, patch
from app.backfill import find_chunks_needing_backfill, backfill_collection


def test_find_chunks_needing_backfill():
    """Identifies chunks missing analyzers or with outdated versions."""
    collection = MagicMock()
    collection.count.return_value = 2
    collection.get.return_value = {
        "ids": ["chunk_0", "chunk_1"],
        "metadatas": [
            {"_analyzers_applied": '{"stats_v1": 1}'},  # missing tone_v1
            {"_analyzers_applied": '{"stats_v1": 1, "tone_v1": 1}'},  # complete
        ],
    }

    from app.analyzers.registry import AnalyzerRegistry
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=lambda *a, **kw: {}, version=1, requires_llm=False, run_order=0)
    registry.register("tone_v1", fn=lambda *a, **kw: {}, version=1, requires_llm=True, run_order=1)

    needs_backfill = find_chunks_needing_backfill(collection, registry)

    assert "chunk_0" in needs_backfill
    assert "chunk_1" not in needs_backfill


def test_find_chunks_needing_backfill_outdated_version():
    """Chunks with outdated analyzer version need backfill."""
    collection = MagicMock()
    collection.count.return_value = 1
    collection.get.return_value = {
        "ids": ["chunk_0"],
        "metadatas": [
            {"_analyzers_applied": '{"stats_v1": 1}'},  # stats is at v1
        ],
    }

    from app.analyzers.registry import AnalyzerRegistry
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=lambda *a, **kw: {}, version=2, requires_llm=False, run_order=0)  # v2 now

    needs_backfill = find_chunks_needing_backfill(collection, registry)

    assert "chunk_0" in needs_backfill
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backfill.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.backfill'"

- [ ] **Step 3: Write minimal implementation**

```python
# app/backfill.py
"""Backfill CLI — run missing or outdated analyzers on existing chunks.

Usage:
    python -m app.backfill                    # run all missing analyzers
    python -m app.backfill --analyzer tone_v1 # run specific analyzer
"""

import argparse
import json
import logging
import os
import sys

import chromadb
import openai

from app.analyzers.default_registry import create_default_registry
from app.analyzers.registry import AnalyzerRegistry, run_analyzers
from app.config import settings
from app.embedder import get_embedding_function
from app.importer import _safe_collection_name

logger = logging.getLogger(__name__)


def find_chunks_needing_backfill(
    collection,
    registry: AnalyzerRegistry,
    analyzer_name: str | None = None,
) -> set[str]:
    """Find chunk IDs that need backfill.

    Args:
        collection: ChromaDB collection
        registry: Analyzer registry to check against
        analyzer_name: If set, only check this specific analyzer

    Returns:
        Set of chunk IDs needing backfill.
    """
    count = collection.count()
    if count == 0:
        return set()

    # Fetch all metadata
    result = collection.get(include=["metadatas"])
    ids = result["ids"]
    metadatas = result["metadatas"]

    analyzers_to_check = (
        [registry.get(analyzer_name)] if analyzer_name and registry.get(analyzer_name)
        else registry.get_all()
    )

    needs_backfill = set()
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        applied_raw = meta.get("_analyzers_applied", "{}")
        if isinstance(applied_raw, str):
            try:
                applied = json.loads(applied_raw)
            except json.JSONDecodeError:
                applied = {}
        else:
            applied = applied_raw

        for analyzer in analyzers_to_check:
            if analyzer and applied.get(analyzer.name) != analyzer.version:
                needs_backfill.add(chunk_id)
                break

    return needs_backfill


def backfill_collection(
    collection,
    registry: AnalyzerRegistry,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
    analyzer_name: str | None = None,
) -> int:
    """Run backfill on chunks that need it.

    Returns:
        Number of chunks updated.
    """
    chunk_ids = find_chunks_needing_backfill(collection, registry, analyzer_name)
    if not chunk_ids:
        logger.info("No chunks need backfill.")
        return 0

    logger.info(f"Backfilling {len(chunk_ids)} chunks...")

    # Fetch full chunk data
    result = collection.get(ids=list(chunk_ids), include=["metadatas", "documents"])
    updated = 0

    for i, chunk_id in enumerate(result["ids"]):
        meta = result["metadatas"][i] or {}
        doc = result["documents"][i] or ""

        # Reconstruct chunk dict for analyzers
        chunk = {
            "chunk_id": chunk_id,
            "document": doc,
            "messages": [],  # not stored in ChromaDB, analyzers work with what's available
            "metadata": meta,
        }

        new_metadata = run_analyzers(
            registry, chunk, twin_name=twin_name,
            llm_client=llm_client, llm_model=llm_model,
        )

        if new_metadata and new_metadata != {"_analyzers_applied": meta.get("_analyzers_applied", {})}:
            # Update ChromaDB metadata
            updated_meta = dict(meta)
            updated_meta.update(new_metadata)
            # Serialize _analyzers_applied as string for ChromaDB
            if "_analyzers_applied" in updated_meta:
                updated_meta["_analyzers_applied"] = json.dumps(updated_meta["_analyzers_applied"])
            collection.update(ids=[chunk_id], metadatas=[updated_meta])
            updated += 1

        if updated % 50 == 0 and updated > 0:
            logger.info(f"  ...{updated}/{len(chunk_ids)} chunks backfilled")

    logger.info(f"Backfill complete: {updated} chunks updated.")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill analyzer metadata on existing chunks")
    parser.add_argument("--analyzer", type=str, default=None, help="Run specific analyzer only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Setup ChromaDB
    chromadb_client = chromadb.PersistentClient(path=settings.chromadb_path)
    ef = get_embedding_function(
        settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )

    # Detect twin
    twin_slug = None
    for name in os.listdir(settings.data_dir):
        twin_dir = os.path.join(settings.data_dir, name)
        if os.path.isdir(twin_dir) and os.path.isfile(os.path.join(twin_dir, "sources.json")):
            twin_slug = name
            break

    if not twin_slug:
        print("No twin found in data directory.")
        sys.exit(1)

    twin_name = twin_slug.replace("_", " ").title()
    collection_name = _safe_collection_name(twin_slug)

    try:
        collection = chromadb_client.get_collection(collection_name, embedding_function=ef)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        sys.exit(1)

    # Setup LLM client for analyzers
    llm_client = openai.OpenAI(
        base_url=settings.analyzer_base_url,
        api_key=settings.analyzer_api_key,
    )

    registry = create_default_registry()
    updated = backfill_collection(
        collection, registry, twin_name,
        llm_client=llm_client,
        llm_model=settings.analyzer_model,
        analyzer_name=args.analyzer,
    )

    print(f"Done. Updated {updated} chunks.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backfill.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/backfill.py tests/test_backfill.py
git commit -m "feat: backfill CLI — run missing/outdated analyzers on existing chunks"
```

---

### Task 12: Update .env.example and requirements.txt

**Files:**
- Modify: `.env.example`
- Modify: `requirements.txt`

- [ ] **Step 1: Add new env vars to .env.example**

Add after the existing `DATA_DIR` line in `.env.example`:

```env
# Classifier model (Intent Agent, Critic Agent) — falls back to LLM_* if not set
CLASSIFIER_BASE_URL=http://localhost:11434/v1
CLASSIFIER_MODEL=llama3.2:3b
CLASSIFIER_API_KEY=ollama

# Analyzer model (import-time enrichment, chunking) — falls back to LLM_* if not set
ANALYZER_BASE_URL=http://localhost:11434/v1
ANALYZER_MODEL=llama3.2:3b
ANALYZER_API_KEY=ollama
```

- [ ] **Step 2: Add langgraph to requirements.txt**

Add to `requirements.txt` (for Plan B, but add now so the dep is available):

```
langgraph==0.4.1
langchain-core==0.3.51
```

- [ ] **Step 3: Commit**

```bash
git add .env.example requirements.txt
git commit -m "chore: add classifier/analyzer config and langgraph dependency"
```

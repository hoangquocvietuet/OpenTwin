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
    chunk: dict
    prev_chunk: dict | None = None
    next_chunk: dict | None = None
    thread_summary: str | None = None


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

    analyzers = registry.get_all()
    current_order = -1
    working_chunk = chunk  # initialize

    for analyzer in analyzers:
        if applied.get(analyzer.name) == analyzer.version:
            continue

        if analyzer.run_order != current_order:
            current_order = analyzer.run_order
            working_metadata = dict(chunk.get("metadata", {}))
            working_metadata.update(all_new_metadata)
            working_chunk = {**chunk, "metadata": working_metadata}

        analyzer_input = AnalyzerInput(
            chunk=working_chunk,
            prev_chunk=prev_chunk,
            next_chunk=next_chunk,
            thread_summary=thread_summary,
        )

        kwargs = {}
        if analyzer.requires_llm:
            kwargs["llm_client"] = llm_client
            kwargs["llm_model"] = llm_model
        kwargs["twin_name"] = twin_name

        result = analyzer.fn(analyzer_input, **kwargs)

        all_new_metadata.update(result)
        applied[analyzer.name] = analyzer.version

    all_new_metadata["_analyzers_applied"] = applied
    return all_new_metadata

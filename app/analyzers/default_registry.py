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

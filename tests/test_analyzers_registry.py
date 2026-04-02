from app.analyzers.registry import AnalyzerInput, AnalyzerRegistry, run_analyzers


def _fake_stats(input: AnalyzerInput, **kwargs) -> dict:
    return {"msg_count": len(input.chunk.get("messages", [])), "language": "vi"}


def _fake_tone(input: AnalyzerInput, **kwargs) -> dict:
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

    assert result == {"_analyzers_applied": {"stats_v1": 1}}


def test_registry_reruns_outdated_version():
    """Registry re-runs analyzers when version is newer."""
    registry = AnalyzerRegistry()
    registry.register("stats_v1", fn=_fake_stats, version=2, requires_llm=False, run_order=0)

    chunk = {
        "messages": [{"author": "A", "text": "hello"}],
        "metadata": {"_analyzers_applied": {"stats_v1": 1}},
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

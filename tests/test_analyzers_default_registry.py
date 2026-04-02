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

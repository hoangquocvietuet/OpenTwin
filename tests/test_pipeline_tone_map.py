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

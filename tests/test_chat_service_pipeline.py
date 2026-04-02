from unittest.mock import MagicMock, patch


def test_chat_uses_pipeline_when_enriched():
    """chat() delegates to pipeline when collection has enriched metadata."""
    with patch("app.chat_service._has_enriched_metadata", return_value=True), \
         patch("app.chat_service._pipeline_chat") as mock_pipeline:

        from app.chat_service import chat, ChatResult
        mock_pipeline.return_value = ChatResult(
            content="pipeline response",
            retrieval_metadata={"chunks": 3, "avg_similarity": 0.8},
            retrieved_chunks=[],
        )

        collection = MagicMock()
        collection.count.return_value = 10
        session_factory = MagicMock()
        session = MagicMock()
        session_factory.return_value.__enter__ = MagicMock(return_value=session)
        session_factory.return_value.__exit__ = MagicMock(return_value=False)

        result = chat(
            content="hello",
            collection=collection,
            session_factory=session_factory,
            twin_slug="test",
            twin_name="Viet",
            system_prompt="You are Viet.",
            rewrite_prompt="Rephrase.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="test",
            llm_api_key="ollama",
        )

        mock_pipeline.assert_called_once()
        assert result.content == "pipeline response"


def test_chat_uses_legacy_when_not_enriched():
    """chat() uses legacy path when collection lacks enriched metadata."""
    with patch("app.chat_service._has_enriched_metadata", return_value=False), \
         patch("app.chat_service._legacy_chat") as mock_legacy:

        from app.chat_service import chat, ChatResult
        mock_legacy.return_value = ChatResult(
            content="legacy response",
            retrieval_metadata={"chunks": 1, "avg_similarity": 0.5},
            retrieved_chunks=[],
        )

        collection = MagicMock()
        collection.count.return_value = 10
        session_factory = MagicMock()
        session = MagicMock()
        session_factory.return_value.__enter__ = MagicMock(return_value=session)
        session_factory.return_value.__exit__ = MagicMock(return_value=False)

        result = chat(
            content="hello",
            collection=collection,
            session_factory=session_factory,
            twin_slug="test",
            twin_name="Viet",
            system_prompt="You are Viet.",
            rewrite_prompt="Rephrase.",
            llm_base_url="http://localhost:11434/v1",
            llm_model="test",
            llm_api_key="ollama",
        )

        mock_legacy.assert_called_once()
        assert result.content == "legacy response"

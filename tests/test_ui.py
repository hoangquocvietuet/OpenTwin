from unittest.mock import MagicMock, patch
from app.ui import create_ui


def test_create_ui_returns_gradio_blocks():
    """create_ui returns a Gradio Blocks instance."""
    import gradio as gr

    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_session_factory = MagicMock()

    ui = create_ui(
        collection=mock_collection,
        session_factory=mock_session_factory,
        twin_slug="hoang_quoc_viet",
        system_prompt="You are Việt.",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        llm_api_key="ollama",
        chromadb_client=MagicMock(),
        data_dir="./data",
        embedding_model="all-MiniLM-L6-v2",
    )
    assert isinstance(ui, gr.Blocks)

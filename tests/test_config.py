# tests/test_config.py
import os
from app.config import Settings


def test_settings_defaults():
    """Settings loads with sensible defaults when no env vars set."""
    settings = Settings()
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.chromadb_path == "./data/chromadb"
    assert settings.sqlite_path == "./db/chat_history.db"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.data_dir == "./data"


def test_settings_from_env(monkeypatch):
    """Settings reads from environment variables."""
    monkeypatch.setenv("LLM_BASE_URL", "http://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")
    settings = Settings()
    assert settings.llm_base_url == "http://example.com/v1"
    assert settings.llm_model == "gpt-4"

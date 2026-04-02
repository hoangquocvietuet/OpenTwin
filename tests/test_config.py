# tests/test_config.py
import os
from app.config import Settings


def test_settings_defaults(monkeypatch):
    """Settings loads with sensible defaults when no env vars set."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("CHROMADB_PATH", raising=False)
    monkeypatch.delenv("SQLITE_PATH", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    settings = Settings()
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.chromadb_path == "./data/chromadb"
    assert settings.sqlite_path == "./db/chat_history.db"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_base_url == "http://localhost:11434/v1"
    assert settings.embedding_api_key == "ollama"
    assert settings.data_dir == "./data"


def test_settings_from_env(monkeypatch):
    """Settings reads from environment variables."""
    monkeypatch.setenv("LLM_BASE_URL", "http://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")
    settings = Settings()
    assert settings.llm_base_url == "http://example.com/v1"
    assert settings.llm_model == "gpt-4"


def test_classifier_config_defaults(monkeypatch):
    """Classifier config falls back to main LLM config."""
    monkeypatch.delenv("CLASSIFIER_BASE_URL", raising=False)
    monkeypatch.delenv("CLASSIFIER_MODEL", raising=False)
    monkeypatch.delenv("CLASSIFIER_API_KEY", raising=False)
    monkeypatch.delenv("ANALYZER_BASE_URL", raising=False)
    monkeypatch.delenv("ANALYZER_MODEL", raising=False)
    monkeypatch.delenv("ANALYZER_API_KEY", raising=False)

    s = Settings()

    assert s.classifier_base_url == s.llm_base_url
    assert s.classifier_model == s.llm_model
    assert s.classifier_api_key == s.llm_api_key
    assert s.analyzer_base_url == s.llm_base_url
    assert s.analyzer_model == s.llm_model
    assert s.analyzer_api_key == s.llm_api_key


def test_classifier_config_overrides(monkeypatch):
    """Classifier config can be overridden via env vars."""
    monkeypatch.setenv("CLASSIFIER_MODEL", "llama3.2:3b")
    monkeypatch.setenv("ANALYZER_MODEL", "phi-3:mini")

    s = Settings()
    assert s.classifier_model == "llama3.2:3b"
    assert s.analyzer_model == "phi-3:mini"

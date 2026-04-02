"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "ollama")
    )
    twin_name: str = field(
        default_factory=lambda: os.getenv("TWIN_NAME", "auto")
    )
    chromadb_path: str = field(
        default_factory=lambda: os.getenv("CHROMADB_PATH", "./data/chromadb")
    )
    sqlite_path: str = field(
        default_factory=lambda: os.getenv("SQLITE_PATH", "./db/chat_history.db")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "ollama")
    )
    data_dir: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "./data")
    )
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


settings = Settings()

"""FastAPI app with Gradio UI mount."""

import logging
import os
import re
import unicodedata

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

import chromadb
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_engine_and_tables, SessionFactory, load_settings as load_db_settings
from app.chat import create_chat_router
from app.api_v2 import create_api_v2_router
from app.embedder import get_embedding_function
from app.importer import _safe_collection_name, rebuild_embeddings
from app.prompt import build_answer_prompt, build_rewrite_prompt, load_fingerprint
from app.sources import (
    load_sources, get_merged_fingerprint_path, get_enabled_chunk_paths,
    migrate_legacy_data,
)
from app.ui import create_ui


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Digital Twins")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Database
    engine = create_engine_and_tables(settings.sqlite_path)
    session_factory = SessionFactory(engine)

    # Load persisted settings from DB (overrides .env defaults)
    db_settings = load_db_settings(session_factory)
    llm_base_url = db_settings.get("llm_base_url", settings.llm_base_url)
    llm_model = db_settings.get("llm_model", settings.llm_model)
    llm_api_key = db_settings.get("llm_api_key", settings.llm_api_key)
    embedding_model = db_settings.get("embedding_model", settings.embedding_model)
    embedding_base_url = db_settings.get("embedding_base_url", settings.embedding_base_url)
    embedding_api_key = db_settings.get("embedding_api_key", settings.embedding_api_key)

    # ChromaDB
    os.makedirs(settings.chromadb_path, exist_ok=True)
    chromadb_client = chromadb.PersistentClient(path=settings.chromadb_path)
    embedding_fn = get_embedding_function(
        embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
    )

    # Detect twin — scan data_dir for directories with sources or legacy data
    twin_slug = None
    system_prompt = "You are a digital twin. Import data to get started."
    rewrite_prompt = "Rephrase the user's message in your style. Output only the rephrased text."
    collection = None
    twin_name = settings.twin_name

    if os.path.isdir(settings.data_dir):
        for name in os.listdir(settings.data_dir):
            twin_dir = os.path.join(settings.data_dir, name)
            if not os.path.isdir(twin_dir):
                continue
            # Check for sources.json or legacy fingerprint
            has_sources = os.path.isfile(os.path.join(twin_dir, "sources.json"))
            has_legacy = os.path.isfile(os.path.join(twin_dir, "style_fingerprint.json"))
            if has_sources or has_legacy:
                twin_slug = name
                break

    if twin_slug:
        # Migrate legacy flat structure if needed
        migrate_legacy_data(settings.data_dir, twin_slug)

        # Load fingerprint from best enabled source
        fp_path = get_merged_fingerprint_path(settings.data_dir, twin_slug)
        if fp_path:
            fingerprint = load_fingerprint(fp_path)
        else:
            fingerprint = None

        if twin_name == "auto":
            twin_name = twin_slug.replace("_", " ").title()

        system_prompt = build_answer_prompt(twin_name, fingerprint)
        rewrite_prompt = build_rewrite_prompt(twin_name, fingerprint)

        collection_name = _safe_collection_name(twin_slug)

        # Get or create collection
        try:
            collection = chromadb_client.get_collection(collection_name, embedding_function=embedding_fn)
        except Exception:
            collection = chromadb_client.get_or_create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_fn,
            )

        # Don't auto-rebuild — user imports sources via the UI
    else:
        twin_slug = "default"
        collection = chromadb_client.get_or_create_collection(
            "default",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,
        )

    # Chat API
    chat_router = create_chat_router(
        collection=collection,
        session_factory=session_factory,
        twin_slug=twin_slug,
        twin_name=twin_name,
        system_prompt=system_prompt,
        rewrite_prompt=rewrite_prompt,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )
    app.include_router(chat_router)

    api_v2_router = create_api_v2_router(
        session_factory=session_factory,
        twin_slug=twin_slug,
        twin_name=twin_name,
        system_prompt=system_prompt,
        rewrite_prompt=rewrite_prompt,
        chromadb_client=chromadb_client,
        data_dir=settings.data_dir,
        collection=collection,
    )
    app.include_router(api_v2_router)

    # Gradio UI
    ui = create_ui(
        collection=collection,
        session_factory=session_factory,
        twin_slug=twin_slug,
        twin_name=twin_name,
        system_prompt=system_prompt,
        rewrite_prompt=rewrite_prompt,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        chromadb_client=chromadb_client,
        data_dir=settings.data_dir,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
    )
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


app = create_app()

if __name__ == "__main__":
    import logging
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)

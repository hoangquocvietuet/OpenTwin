"""FastAPI app with Gradio UI mount."""

import os
import re
import unicodedata

import chromadb
import gradio as gr
from fastapi import FastAPI

from app.config import settings
from app.database import create_engine_and_tables, SessionFactory
from app.chat import create_chat_router
from app.embedder import get_embedding_function
from app.prompt import build_system_prompt, load_fingerprint
from app.ui import create_ui


def _safe_collection_name(name: str) -> str:
    """Convert a name to a ChromaDB-safe collection name (ASCII alphanumeric + _ -)."""
    # Normalize unicode to ASCII equivalents where possible
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace spaces with underscores, strip anything not alphanumeric/underscore/hyphen
    ascii_name = ascii_name.replace(" ", "_")
    ascii_name = re.sub(r"[^a-zA-Z0-9_\-]", "", ascii_name)
    ascii_name = ascii_name.strip("_-")
    return ascii_name or "twin"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Digital Twins")

    # Database — session factory for thread-safe access
    engine = create_engine_and_tables(settings.sqlite_path)
    session_factory = SessionFactory(engine)

    # ChromaDB with explicit embedding model
    os.makedirs(settings.chromadb_path, exist_ok=True)
    chromadb_client = chromadb.PersistentClient(path=settings.chromadb_path)
    embedding_fn = get_embedding_function(settings.embedding_model)

    # Detect twin — scan data_dir for directories with a fingerprint
    twin_slug = None
    system_prompt = "You are a digital twin. Import data to get started."
    collection = None

    if os.path.isdir(settings.data_dir):
        for name in os.listdir(settings.data_dir):
            fp_path = os.path.join(settings.data_dir, name, "style_fingerprint.json")
            if os.path.isfile(fp_path):
                twin_slug = name
                break

    if twin_slug:
        fp_path = os.path.join(settings.data_dir, twin_slug, "style_fingerprint.json")
        fingerprint = load_fingerprint(fp_path)

        # Determine display name
        twin_name = settings.twin_name
        if twin_name == "auto":
            twin_name = twin_slug.replace("_", " ").title()

        system_prompt = build_system_prompt(twin_name, fingerprint)

        # ChromaDB collection names must be ASCII alphanumeric
        collection_name = _safe_collection_name(twin_slug)

        # Get or create ChromaDB collection with explicit embedding function
        try:
            collection = chromadb_client.get_collection(collection_name, embedding_function=embedding_fn)
        except Exception:
            collection = chromadb_client.get_or_create_collection(
                collection_name, embedding_function=embedding_fn
            )

        # If collection is empty but chunks exist, auto-embed
        if collection.count() == 0:
            chunks_path = os.path.join(settings.data_dir, twin_slug, "train_chunks.jsonl")
            if os.path.isfile(chunks_path):
                from app.embedder import load_chunks_from_jsonl, ingest_chunks
                chunks = load_chunks_from_jsonl(chunks_path)
                collection = ingest_chunks(
                    chromadb_client, collection_name, chunks, embedding_function=embedding_fn
                )
    else:
        twin_slug = "default"
        collection = chromadb_client.get_or_create_collection(
            "default", embedding_function=embedding_fn
        )

    # Chat API + Export API
    chat_router = create_chat_router(
        collection=collection,
        session_factory=session_factory,
        twin_slug=twin_slug,
        system_prompt=system_prompt,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
    )
    app.include_router(chat_router)

    # Gradio UI
    ui = create_ui(
        collection=collection,
        session_factory=session_factory,
        twin_slug=twin_slug,
        system_prompt=system_prompt,
        llm_base_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
        chromadb_client=chromadb_client,
        data_dir=settings.data_dir,
        embedding_model=settings.embedding_model,
    )
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)

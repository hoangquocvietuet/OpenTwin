import os
import sqlite3
import sys
from pathlib import Path

import chromadb

# Ensure repo root is importable so `import app.*` works when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.embedder import get_embedding_function  # noqa: E402


def main() -> int:
    chroma_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
    sqlite_path = os.getenv("SQLITE_PATH", "./db/chat_history.db")
    twin_slug = os.getenv("TWIN_SLUG") or ""
    query = os.getenv("QUERY", "hello")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "ollama")

    # Match app behavior: persisted DB settings override env defaults.
    if os.path.isfile(sqlite_path):
        try:
            with sqlite3.connect(sqlite_path) as conn:
                rows = conn.execute(
                    "SELECT key, value FROM app_settings WHERE key IN (?, ?, ?)",
                    ("embedding_model", "embedding_base_url", "embedding_api_key"),
                ).fetchall()
            persisted = {k: v for k, v in rows}
            embedding_model = persisted.get("embedding_model", embedding_model)
            embedding_base_url = persisted.get("embedding_base_url", embedding_base_url)
            embedding_api_key = persisted.get("embedding_api_key", embedding_api_key)
        except Exception:
            # Best effort only. Keep env values if DB access fails.
            pass

    client = chromadb.PersistentClient(path=chroma_path)
    embedding_fn = get_embedding_function(
        embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
    )

    print(f"CHROMA_PATH={chroma_path}")
    print(f"SQLITE_PATH={sqlite_path}")
    print(f"EMBEDDING_MODEL={embedding_model}")
    print(f"EMBEDDING_BASE_URL={embedding_base_url}")
    print("COLLECTIONS:")
    cols = client.list_collections()
    col_names: list[str] = []
    for c in cols:
        # Chroma v0.6 returns names (strings). Older versions may return objects/dicts.
        if isinstance(c, str):
            name = c
        elif isinstance(c, dict):
            name = c.get("name") or str(c)
        else:
            # Some Chroma objects raise on attribute access, so don't touch c.name.
            name = str(c)

        col_names.append(name)
        print(f"- {name}")

    if not col_names:
        print("\nNo collections found. This usually means nothing has been embedded yet.")
        return 3

    # Try exact name if provided. If not provided, auto-select if unambiguous.
    try:
        if twin_slug:
            col = client.get_collection(twin_slug, embedding_function=embedding_fn)
            col_name = twin_slug
        elif len(col_names) == 1:
            col_name = col_names[0]
            col = client.get_collection(col_name, embedding_function=embedding_fn)
        else:
            print("\nMultiple collections found.")
            print("Re-run with one of these names, for example:")
            print(f'  QUERY="hello" TWIN_SLUG="{col_names[0]}" python3 scripts/inspect_chroma.py')
            return 2
    except Exception:
        print(f"\nCould not open collection {twin_slug!r}.")
        print("Re-run with TWIN_SLUG set to one of the collection names above.")
        return 2

    print(f"\nCOLLECTION={col_name}")
    print(f"COUNT={col.count()}")

    # Peek a few docs + metadatas
    peek = col.peek()
    ids = peek.get("ids", [])[:3]
    print("\nPEEK (first 3):")
    for i, _id in enumerate(ids):
        doc = (peek.get("documents") or [""])[i]
        meta = (peek.get("metadatas") or [{}])[i]
        print(f"- id={_id}")
        print(f"  meta_keys={sorted(list(meta.keys()))}")
        print(f"  doc_preview={repr((doc or '')[:140])}")

    # Run a query and print distances so you can see if the distance<0.8 filter
    # is likely to remove everything.
    print(f"\nQUERY={repr(query)}")
    res = col.query(
        query_texts=[query],
        n_results=10,
        include=["distances", "documents", "metadatas"],
    )
    distances = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    print("TOP 10 distances:")
    for i, d in enumerate(distances):
        meta = metas[i] if i < len(metas) else {}
        score = meta.get("score")
        chunk_type = meta.get("chunk_type")
        preview = (docs[i] or "")[:90].replace("\n", " ")
        print(f"- {i+1:02d}. distance={d:.4f} sim≈{(1-d):.4f} score={score} type={chunk_type} preview={repr(preview)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


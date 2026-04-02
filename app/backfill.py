"""Backfill CLI — run missing or outdated analyzers on existing chunks.

Usage:
    python -m app.backfill                    # run all missing analyzers
    python -m app.backfill --analyzer tone_v1 # run specific analyzer
"""

import argparse
import json
import logging
import os
import sys

from app.analyzers.registry import AnalyzerRegistry, run_analyzers

logger = logging.getLogger(__name__)


def find_chunks_needing_backfill(
    collection,
    registry: AnalyzerRegistry,
    analyzer_name: str | None = None,
) -> set[str]:
    """Find chunk IDs that need backfill.

    Args:
        collection: ChromaDB collection
        registry: Analyzer registry to check against
        analyzer_name: If set, only check this specific analyzer

    Returns:
        Set of chunk IDs needing backfill.
    """
    count = collection.count()
    if count == 0:
        return set()

    analyzers_to_check = (
        [registry.get(analyzer_name)] if analyzer_name and registry.get(analyzer_name)
        else registry.get_all()
    )

    needs_backfill = set()
    batch_size = 1000
    for offset in range(0, count, batch_size):
        result = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        ids = result["ids"]
        metadatas = result["metadatas"]

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] or {}
            applied_raw = meta.get("_analyzers_applied", "{}")
            if isinstance(applied_raw, str):
                try:
                    applied = json.loads(applied_raw)
                except json.JSONDecodeError:
                    applied = {}
            else:
                applied = applied_raw

            for analyzer in analyzers_to_check:
                if analyzer and applied.get(analyzer.name) != analyzer.version:
                    needs_backfill.add(chunk_id)
                    break

    return needs_backfill


def backfill_collection(
    collection,
    registry: AnalyzerRegistry,
    twin_name: str,
    llm_client=None,
    llm_model: str | None = None,
    analyzer_name: str | None = None,
) -> int:
    """Run backfill on chunks that need it.

    Returns:
        Number of chunks updated.
    """
    # Only run non-LLM analyzers during backfill (messages aren't stored in ChromaDB,
    # so LLM analyzers would just produce defaults and permanently mark as "applied")
    from app.analyzers.registry import AnalyzerRegistry as _AR
    backfill_registry = _AR()
    for analyzer in registry.get_all():
        if not analyzer.requires_llm:
            backfill_registry.register(
                analyzer.name, fn=analyzer.fn, version=analyzer.version,
                requires_llm=analyzer.requires_llm, run_order=analyzer.run_order,
            )

    # Use the filtered registry for both detection and execution
    chunk_ids = find_chunks_needing_backfill(collection, backfill_registry, analyzer_name)
    if not chunk_ids:
        logger.info("No chunks need backfill.")
        return 0

    total = len(chunk_ids)
    logger.info(f"Backfilling {total} chunks...")

    updated = 0
    failed = 0
    batch_size = 100
    chunk_id_list = list(chunk_ids)

    for batch_start in range(0, total, batch_size):
        batch_ids = chunk_id_list[batch_start:batch_start + batch_size]
        result = collection.get(ids=batch_ids, include=["metadatas", "documents"])

        update_ids = []
        update_metas = []

        for i, chunk_id in enumerate(result["ids"]):
            try:
                meta = result["metadatas"][i] or {}
                doc = result["documents"][i] or ""

                chunk = {
                    "chunk_id": chunk_id,
                    "document": doc,
                    "messages": [],
                    "metadata": meta,
                }

                new_metadata = run_analyzers(
                    backfill_registry, chunk, twin_name=twin_name,
                    llm_client=llm_client, llm_model=llm_model,
                )

                if new_metadata and new_metadata != {"_analyzers_applied": meta.get("_analyzers_applied", {})}:
                    updated_meta = dict(meta)
                    updated_meta.update(new_metadata)
                    if "_analyzers_applied" in updated_meta and isinstance(updated_meta["_analyzers_applied"], dict):
                        updated_meta["_analyzers_applied"] = json.dumps(updated_meta["_analyzers_applied"])
                    update_ids.append(chunk_id)
                    update_metas.append(updated_meta)
            except Exception:
                logger.warning(f"Failed to backfill chunk {chunk_id}", exc_info=True)
                failed += 1

        if update_ids:
            collection.update(ids=update_ids, metadatas=update_metas)
            updated += len(update_ids)

        if updated > 0 and updated % 500 < batch_size:
            logger.info(f"  ...{updated}/{total} chunks backfilled")

    logger.info(f"Backfill complete: {updated} chunks updated, {failed} failed.")
    return updated


def main():
    import chromadb
    import openai

    from app.analyzers.default_registry import create_default_registry
    from app.config import settings
    from app.embedder import get_embedding_function
    from app.importer import _safe_collection_name

    parser = argparse.ArgumentParser(description="Backfill analyzer metadata on existing chunks")
    parser.add_argument("--analyzer", type=str, default=None, help="Run specific analyzer only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    chromadb_client = chromadb.PersistentClient(path=settings.chromadb_path)
    ef = get_embedding_function(
        settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )

    twin_slug = None
    for name in os.listdir(settings.data_dir):
        twin_dir = os.path.join(settings.data_dir, name)
        if os.path.isdir(twin_dir) and os.path.isfile(os.path.join(twin_dir, "sources.json")):
            twin_slug = name
            break

    if not twin_slug:
        print("No twin found in data directory.")
        sys.exit(1)

    twin_name = twin_slug.replace("_", " ").title()
    collection_name = _safe_collection_name(twin_slug)

    try:
        collection = chromadb_client.get_collection(collection_name, embedding_function=ef)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        sys.exit(1)

    llm_client = openai.OpenAI(
        base_url=settings.analyzer_base_url,
        api_key=settings.analyzer_api_key,
    )

    registry = create_default_registry()
    updated = backfill_collection(
        collection, registry, twin_name,
        llm_client=llm_client,
        llm_model=settings.analyzer_model,
        analyzer_name=args.analyzer,
    )

    print(f"Done. Updated {updated} chunks.")


if __name__ == "__main__":
    main()

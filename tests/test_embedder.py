import chromadb
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function, OpenAICompatibleEmbeddingFunction


def test_get_embedding_function():
    """Returns an OpenAICompatibleEmbeddingFunction."""
    ef = get_embedding_function("text-embedding-3-small")
    assert isinstance(ef, OpenAICompatibleEmbeddingFunction)


def test_load_chunks_from_jsonl(sample_chunks_path):
    """Loads chunks from JSONL file."""
    chunks = load_chunks_from_jsonl(sample_chunks_path)
    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "dm_test_0"
    assert chunks[0]["response"]["text"] == "đang code dự án mới nè"


def test_ingest_chunks_creates_collection(sample_chunks_path, tmp_path):
    """Ingesting chunks creates a ChromaDB collection with documents."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")

    chunks = load_chunks_from_jsonl(sample_chunks_path)
    collection = ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)

    assert collection.count() == 3
    # Verify we can query
    results = collection.query(query_texts=["code dự án"], n_results=2)
    assert len(results["ids"][0]) == 2


def test_ingest_chunks_stores_metadata(sample_chunks_path, tmp_path):
    """Chunk metadata is stored in ChromaDB."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")

    chunks = load_chunks_from_jsonl(sample_chunks_path)
    collection = ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)

    result = collection.get(ids=["dm_test_0"], include=["metadatas", "documents"])
    meta = result["metadatas"][0]
    assert meta["chunk_type"] == "dm"
    assert meta["score"] == 1.5
    assert meta["context_turns"] == 1
    assert meta["response_length"] == 23
    # document is the response text
    assert "đang code dự án mới nè" in result["documents"][0]


def test_ingest_overwrites_existing_collection(sample_chunks_path, tmp_path):
    """Re-ingesting deletes the old collection and creates a new one."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")

    chunks = load_chunks_from_jsonl(sample_chunks_path)
    ingest_chunks(client, "hoang_quoc_viet", chunks, embedding_function=ef)
    assert client.get_collection("hoang_quoc_viet", embedding_function=ef).count() == 3

    # Re-ingest with only 2 chunks
    ingest_chunks(client, "hoang_quoc_viet", chunks[:2], embedding_function=ef)
    assert client.get_collection("hoang_quoc_viet", embedding_function=ef).count() == 2

import chromadb
from app.embedder import ingest_chunks, load_chunks_from_jsonl, get_embedding_function
from app.retrieval import retrieve_chunks, format_few_shot_examples


def test_retrieve_chunks_returns_results(sample_chunks_path, tmp_path):
    """Retrieval returns ranked chunks for a query."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    ef = get_embedding_function("all-MiniLM-L6-v2")
    chunks = load_chunks_from_jsonl(sample_chunks_path)
    collection = ingest_chunks(client, "test_twin", chunks, embedding_function=ef)

    results = retrieve_chunks(collection, "bạn đang làm gì", n_results=2)
    assert len(results) == 2
    assert "chunk_id" in results[0]
    assert "document" in results[0]
    assert "distance" in results[0]
    assert "metadata" in results[0]


def test_retrieve_chunks_empty_collection(tmp_path):
    """Retrieval on empty collection returns empty list."""
    chromadb_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection("empty_twin")

    results = retrieve_chunks(collection, "hello", n_results=5)
    assert results == []


def test_format_few_shot_examples():
    """Formats retrieved chunks as few-shot examples for the prompt."""
    retrieved = [
        {
            "chunk_id": "dm_test_0",
            "document": "Friend: bạn làm gì đấy?\nViệt: đang code dự án mới nè",
            "distance": 0.3,
            "metadata": {"chunk_type": "dm", "score": 1.5},
        },
        {
            "chunk_id": "dm_test_1",
            "document": "Friend: đi ăn không?\nViệt: ok đi anh",
            "distance": 0.5,
            "metadata": {"chunk_type": "dm", "score": 1.2},
        },
    ]
    examples = format_few_shot_examples(retrieved)
    assert "bạn làm gì đấy?" in examples
    assert "đang code dự án mới nè" in examples
    assert "đi ăn không?" in examples

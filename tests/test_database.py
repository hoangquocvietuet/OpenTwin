import tempfile
import os
from app.database import create_engine_and_tables, ChatMessage, SessionFactory


def test_create_tables_and_insert(tmp_path):
    """Can create tables and insert a chat message."""
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    factory = SessionFactory(engine)

    with factory() as session:
        msg = ChatMessage(
            twin_slug="hoang_quoc_viet",
            role="user",
            content="xin chào",
        )
        session.add(msg)
        session.commit()

        result = session.query(ChatMessage).first()
        assert result.content == "xin chào"
        assert result.role == "user"
        assert result.twin_slug == "hoang_quoc_viet"
        assert result.id == 1
        assert result.created_at is not None


def test_retrieval_metadata_json(tmp_path):
    """retrieval_metadata stores JSON data."""
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    factory = SessionFactory(engine)

    with factory() as session:
        msg = ChatMessage(
            twin_slug="hoang_quoc_viet",
            role="assistant",
            content="đang code nè",
            retrieval_metadata={"chunks": 3, "avg_similarity": 0.85},
            tokens_used=42,
        )
        session.add(msg)
        session.commit()

        result = session.query(ChatMessage).first()
        assert result.retrieval_metadata["chunks"] == 3
        assert result.tokens_used == 42


def test_get_recent_messages(tmp_path):
    """Can query recent messages in chronological order."""
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    factory = SessionFactory(engine)

    with factory() as session:
        for i in range(15):
            session.add(ChatMessage(
                twin_slug="hoang_quoc_viet",
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
            ))
        session.commit()

        recent = (
            session.query(ChatMessage)
            .filter_by(twin_slug="hoang_quoc_viet")
            .order_by(ChatMessage.id.desc())
            .limit(10)
            .all()
        )
        recent.reverse()
        assert len(recent) == 10
        assert recent[0].content == "message 5"
        assert recent[-1].content == "message 14"

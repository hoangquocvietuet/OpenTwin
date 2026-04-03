"""Tests for Conversation model and conversation_id FK on ChatMessage."""

import uuid
from datetime import datetime, timezone

import pytest

from app.database import (
    Base, Conversation, ChatMessage,
    create_engine_and_tables, SessionFactory,
)


@pytest.fixture
def session_factory(tmp_path):
    db_path = str(tmp_path / "test.db")
    engine = create_engine_and_tables(db_path)
    return SessionFactory(engine)


def test_create_conversation(session_factory):
    conv_id = str(uuid.uuid4())
    with session_factory() as session:
        conv = Conversation(id=conv_id, twin_slug="test_twin", title="Hello")
        session.add(conv)
        session.commit()

    with session_factory() as session:
        result = session.query(Conversation).filter_by(id=conv_id).first()
        assert result is not None
        assert result.twin_slug == "test_twin"
        assert result.title == "Hello"
        assert result.created_at is not None
        assert result.updated_at is not None


def test_chat_message_with_conversation_id(session_factory):
    conv_id = str(uuid.uuid4())
    with session_factory() as session:
        conv = Conversation(id=conv_id, twin_slug="test_twin", title="Test")
        session.add(conv)
        session.commit()

    with session_factory() as session:
        msg = ChatMessage(
            twin_slug="test_twin",
            role="user",
            content="hello",
            conversation_id=conv_id,
        )
        session.add(msg)
        session.commit()

    with session_factory() as session:
        msg = session.query(ChatMessage).first()
        assert msg.conversation_id == conv_id


def test_chat_message_without_conversation_id(session_factory):
    """conversation_id is nullable for backward compat."""
    with session_factory() as session:
        msg = ChatMessage(twin_slug="test_twin", role="user", content="hello")
        session.add(msg)
        session.commit()

    with session_factory() as session:
        msg = session.query(ChatMessage).first()
        assert msg.conversation_id is None


def test_delete_conversation_cascades_messages(session_factory):
    conv_id = str(uuid.uuid4())
    with session_factory() as session:
        conv = Conversation(id=conv_id, twin_slug="test_twin", title="Test")
        session.add(conv)
        session.add(ChatMessage(twin_slug="test_twin", role="user", content="hi", conversation_id=conv_id))
        session.add(ChatMessage(twin_slug="test_twin", role="assistant", content="hello", conversation_id=conv_id))
        session.commit()

    with session_factory() as session:
        conv = session.query(Conversation).filter_by(id=conv_id).first()
        session.delete(conv)
        session.commit()

    with session_factory() as session:
        assert session.query(ChatMessage).filter_by(conversation_id=conv_id).count() == 0


def test_list_conversations_with_last_message(session_factory):
    """Verify we can query conversations with their last message in one query."""
    from sqlalchemy import func

    conv1_id = str(uuid.uuid4())
    conv2_id = str(uuid.uuid4())
    with session_factory() as session:
        session.add(Conversation(id=conv1_id, twin_slug="t", title="Conv 1"))
        session.add(Conversation(id=conv2_id, twin_slug="t", title="Conv 2"))
        session.add(ChatMessage(twin_slug="t", role="user", content="first", conversation_id=conv1_id))
        session.add(ChatMessage(twin_slug="t", role="user", content="second", conversation_id=conv1_id))
        session.add(ChatMessage(twin_slug="t", role="user", content="only", conversation_id=conv2_id))
        session.commit()

    with session_factory() as session:
        # Subquery for max message id per conversation
        last_msg = (
            session.query(
                ChatMessage.conversation_id,
                func.max(ChatMessage.id).label("max_id"),
            )
            .group_by(ChatMessage.conversation_id)
            .subquery()
        )
        results = (
            session.query(Conversation, ChatMessage.content)
            .join(last_msg, Conversation.id == last_msg.c.conversation_id)
            .join(ChatMessage, ChatMessage.id == last_msg.c.max_id)
            .filter(Conversation.twin_slug == "t")
            .order_by(Conversation.updated_at.desc())
            .all()
        )
        assert len(results) == 2

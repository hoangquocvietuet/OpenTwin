"""SQLite database setup for chat history.

Uses a session factory pattern for thread-safety with uvicorn's thread pool.
Each request/operation creates its own session via the factory.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import sqlite3

from sqlalchemy import create_engine, event, Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

Base = declarative_base()


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    twin_slug = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    retrieval_metadata = Column(JSON, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True, index=True)
    conversation = relationship("Conversation", back_populates="messages")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    twin_slug = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False, default="New Chat")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")


class AppSetting(Base):
    __tablename__ = "app_settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)


def _migrate_add_conversation_id(engine):
    """Add conversation_id column to chat_messages if missing (one-time migration)."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("chat_messages")]
    if "conversation_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE chat_messages ADD COLUMN conversation_id VARCHAR REFERENCES conversations(id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_conversation_id ON chat_messages (conversation_id)"
            ))


def create_engine_and_tables(db_path: str):
    """Create SQLite engine and ensure tables exist."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    # Migrate existing DBs that predate the Conversation model
    try:
        _migrate_add_conversation_id(engine)
    except Exception:
        pass  # Table might not exist yet (fresh DB)
    return engine


class SessionFactory:
    """Thread-safe session factory. Creates a new session per call.

    Usage:
        factory = SessionFactory(engine)
        with factory() as session:
            session.query(...)
    """

    def __init__(self, engine):
        self._sessionmaker = sessionmaker(bind=engine)

    @contextmanager
    def __call__(self):
        session = self._sessionmaker()
        try:
            yield session
        finally:
            session.close()


def load_settings(session_factory: "SessionFactory") -> dict[str, str]:
    """Load all persisted settings from the database."""
    with session_factory() as session:
        rows = session.query(AppSetting).all()
        return {row.key: row.value for row in rows}


def save_settings(session_factory: "SessionFactory", settings: dict[str, str]):
    """Persist settings to the database (upsert)."""
    with session_factory() as session:
        for key, value in settings.items():
            existing = session.query(AppSetting).filter_by(key=key).first()
            if existing:
                existing.value = value
            else:
                session.add(AppSetting(key=key, value=value))
        session.commit()

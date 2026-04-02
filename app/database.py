"""SQLite database setup for chat history.

Uses a session factory pattern for thread-safety with uvicorn's thread pool.
Each request/operation creates its own session via the factory.
"""

import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    twin_slug = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    retrieval_metadata = Column(JSON, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_engine_and_tables(db_path: str):
    """Create SQLite engine and ensure tables exist."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
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

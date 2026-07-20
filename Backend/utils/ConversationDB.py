"""
Mosaic Conversation Database

Supports both SQLite (local dev) and PostgreSQL (production).
Controlled by DATABASE_URL environment variable:
  - sqlite:///conversations.db  (default, local)
  - postgresql://user:pass@host:5432/mosaic  (production)
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import QueuePool, StaticPool
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

Base = declarative_base()


class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(String(100), nullable=True, index=True)
    conversation_data = Column(JSON, nullable=True)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    # Index for fast user queries
    __table_args__ = (
        Index('ix_conversations_user_updated', 'user_id', 'updated_at'),
    )


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent = Column(String(100), nullable=True)
    message_data = Column(JSON, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index('ix_messages_conv_time', 'conversation_id', 'timestamp'),
    )


class ConversationManager:
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the conversation database.
        
        Args:
            db_url: SQLAlchemy connection string. Defaults to DATABASE_URL env var,
                    falling back to local SQLite.
        """
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///conversations.db")

        # Configure engine based on database type
        if self.db_url.startswith("sqlite"):
            # SQLite: use StaticPool for single-connection safety
            self.engine = create_engine(
                self.db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            # PostgreSQL/MySQL: use connection pooling
            self.engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Check connection health before use
                echo=False,
            )

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """Get a database session as a context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_conversation(self, title: str, user_id: Optional[str] = None, conversation_data: Optional[Dict] = None) -> Any:
        """Create a new conversation."""
        with self.get_session() as session:
            conversation = Conversation(
                title=title,
                user_id=user_id,
                conversation_data=conversation_data or {}
            )
            session.add(conversation)
            session.flush()
            session.refresh(conversation)
            conv_id = conversation.id
            conv_title = conversation.title
            conv_created = conversation.created_at
            conv_updated = conversation.updated_at

        return type('Conversation', (), {
            'id': conv_id, 'title': conv_title,
            'created_at': conv_created, 'updated_at': conv_updated
        })()

    def add_message(self, conversation_id: int, role: str, content: str,
                   agent: Optional[str] = None, message_data: Optional[Dict] = None) -> Any:
        """Add a message to a conversation."""
        with self.get_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                agent=agent,
                message_data=message_data or {}
            )
            session.add(message)

            conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conversation:
                conversation.updated_at = datetime.utcnow()

            session.flush()
            session.refresh(message)
            return message

    def get_conversation(self, conversation_id: int) -> Optional[Any]:
        """Get a conversation by ID."""
        with self.get_session() as session:
            convo = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not convo:
                return None
            return type('Conversation', (), {
                'id': convo.id, 'title': convo.title,
                'created_at': convo.created_at, 'updated_at': convo.updated_at,
                'user_id': convo.user_id
            })()

    def get_conversations(self, user_id: Optional[str] = None, limit: int = 50) -> List[Any]:
        """Get all conversations, optionally filtered by user_id."""
        with self.get_session() as session:
            query = session.query(Conversation).order_by(Conversation.updated_at.desc())
            if user_id:
                query = query.filter(Conversation.user_id == user_id)
            results = query.limit(limit).all()
            return [
                type('Conversation', (), {
                    'id': c.id, 'title': c.title,
                    'created_at': c.created_at, 'updated_at': c.updated_at
                })()
                for c in results
            ]

    def get_messages(self, conversation_id: int, limit: Optional[int] = None) -> List[Any]:
        """Get messages for a conversation, optionally limited."""
        with self.get_session() as session:
            query = session.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc())
            if limit:
                query = query.limit(limit)
            results = query.all()
            return [
                type('Message', (), {
                    'id': m.id, 'role': m.role, 'content': m.content,
                    'agent': m.agent, 'timestamp': m.timestamp,
                    'conversation_id': m.conversation_id
                })()
                for m in results
            ]

    def get_conversation_context(self, conversation_id: int, max_messages: int = 20) -> List[Dict[str, Any]]:
        """Get conversation context formatted for the LLM."""
        messages = self.get_messages(conversation_id, limit=max_messages)
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def update_conversation_title(self, conversation_id: int, new_title: str) -> bool:
        """Update a conversation's title."""
        with self.get_session() as session:
            conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conversation:
                conversation.title = new_title
                conversation.updated_at = datetime.utcnow()
                return True
            return False

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation and all its messages."""
        with self.get_session() as session:
            conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conversation:
                session.delete(conversation)
                return True
            return False

    def search_conversations(self, query: str, user_id: Optional[str] = None) -> List[Any]:
        """Search conversations by title."""
        with self.get_session() as session:
            q = session.query(Conversation).filter(Conversation.title.contains(query))
            if user_id:
                q = q.filter(Conversation.user_id == user_id)
            results = q.order_by(Conversation.updated_at.desc()).all()
            return [
                type('Conversation', (), {
                    'id': c.id, 'title': c.title,
                    'created_at': c.created_at, 'updated_at': c.updated_at
                })()
                for c in results
            ]

    def get_conversation_stats(self, conversation_id: int) -> Dict[str, Any]:
        """Get statistics for a conversation."""
        with self.get_session() as session:
            messages = session.query(Message).filter(Message.conversation_id == conversation_id).all()
            return {
                "total_messages": len(messages),
                "user_messages": len([m for m in messages if m.role == "user"]),
                "assistant_messages": len([m for m in messages if m.role == "assistant"]),
                "first_message": str(min(messages, key=lambda x: x.timestamp).timestamp) if messages else None,
                "last_message": str(max(messages, key=lambda x: x.timestamp).timestamp) if messages else None,
            }

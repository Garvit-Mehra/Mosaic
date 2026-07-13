from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(String(100), nullable=True)  # For future multi-user support
    conversation_data = Column(JSON, nullable=True)  # For additional conversation data
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title='{self.title}', created_at={self.created_at})>"

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    role = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent = Column(String(100), nullable=True)  # Which agent handled this message
    message_data = Column(JSON, nullable=True)  # For additional message data
    
    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"

class ConversationManager:
    def __init__(self, db_path: str = "conversations.db"):
        """Initialize the conversation database manager."""
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
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
    
    def create_conversation(self, title: str, user_id: Optional[str] = None, conversation_data: Optional[Dict] = None) -> Conversation:
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
            # Detach for use outside session
            conv_id = conversation.id
            conv_title = conversation.title
            conv_created = conversation.created_at
            conv_updated = conversation.updated_at
        
        # Return a simple object with the data
        conv = type('Conversation', (), {
            'id': conv_id, 'title': conv_title,
            'created_at': conv_created, 'updated_at': conv_updated
        })()
        return conv
    
    def add_message(self, conversation_id: int, role: str, content: str, 
                   agent: Optional[str] = None, message_data: Optional[Dict] = None) -> Message:
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
            
            # Update conversation's updated_at timestamp
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
        """
        Get conversation context formatted for the LLM.
        Returns messages in the format expected by the agent.
        """
        messages = self.get_messages(conversation_id, limit=max_messages)
        return [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]
    
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
        """Search conversations by title or content."""
        with self.get_session() as session:
            # Search in conversation titles
            title_matches = session.query(Conversation).filter(
                Conversation.title.contains(query)
            )
            if user_id:
                title_matches = title_matches.filter(Conversation.user_id == user_id)
            
            # Search in message content
            content_matches = session.query(Conversation).join(Message).filter(
                Message.content.contains(query)
            )
            if user_id:
                content_matches = content_matches.filter(Conversation.user_id == user_id)
            
            # Combine and deduplicate results
            all_matches = title_matches.union(content_matches).order_by(Conversation.updated_at.desc())
            results = all_matches.all()
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
            
            stats = {
                "total_messages": len(messages),
                "user_messages": len([m for m in messages if m.role == "user"]),
                "assistant_messages": len([m for m in messages if m.role == "assistant"]),
                "system_messages": len([m for m in messages if m.role == "system"]),
                "first_message": str(min(messages, key=lambda x: x.timestamp).timestamp) if messages else None,
                "last_message": str(max(messages, key=lambda x: x.timestamp).timestamp) if messages else None,
            }
            
            return stats 
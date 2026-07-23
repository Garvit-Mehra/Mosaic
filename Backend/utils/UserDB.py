"""
Mosaic User Database

Stores users with:
- username (unique)
- email (unique)
- password_hash (bcrypt)
- role (admin/user)
- verified (email verification status)
- created_at

Uses the same DATABASE_URL as ConversationDB — shares the database.
"""

import os
from datetime import datetime
from typing import Optional, List, Any

import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_users_email_verified', 'email', 'verified'),
    )


class UserMCPServer(Base):
    __tablename__ = 'user_mcp_servers'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=False, default="")
    url = Column(String(500), nullable=False)
    transport = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_user_mcp_user_name', 'user_id', 'name', unique=True),
    )


class UserManager:
    """Manages user CRUD operations against the database."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///users.db")

        # If a raw path is passed (no ://), assume SQLite
        if "://" not in self.db_url:
            self.db_url = f"sqlite:///{self.db_url}"

        if self.db_url.startswith("sqlite"):
            self.engine = create_engine(
                self.db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            self.engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_user(self, username: str, email: str, password: str, role: str = "user", verified: bool = False) -> dict:
        """
        Create a new user. Password is hashed with bcrypt.
        Returns the user dict or raises ValueError if username/email taken.
        """
        import re

        # Validate username: alphanumeric + underscores only, 3-50 chars
        if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
            raise ValueError("Username must be 3-50 characters, alphanumeric and underscores only.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")
        if "@" not in email:
            raise ValueError("Invalid email address.")

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        with self.get_session() as session:
            # Check uniqueness
            existing = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            if existing:
                if existing.username == username:
                    raise ValueError(f"Username '{username}' is already taken.")
                else:
                    raise ValueError(f"Email '{email}' is already registered.")

            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                verified=verified,
            )
            session.add(user)
            session.flush()
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "verified": user.verified,
            }

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get a user by username."""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return None
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "password_hash": user.password_hash,
                "role": user.role,
                "verified": user.verified,
                "created_at": str(user.created_at),
            }

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get a user by email."""
        with self.get_session() as session:
            user = session.query(User).filter(User.email == email).first()
            if not user:
                return None
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "password_hash": user.password_hash,
                "role": user.role,
                "verified": user.verified,
                "created_at": str(user.created_at),
            }

    def list_users(self) -> List[dict]:
        """List all users (without password hashes)."""
        with self.get_session() as session:
            users = session.query(User).order_by(User.created_at.desc()).all()
            return [
                {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role,
                    "verified": u.verified,
                    "created_at": str(u.created_at),
                }
                for u in users
            ]

    def update_user(self, username: str, **kwargs) -> bool:
        """
        Update user fields. Supported: email, role, verified, password.
        If 'password' is provided, it will be hashed.
        """
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return False

            if "email" in kwargs:
                user.email = kwargs["email"]
            if "role" in kwargs:
                user.role = kwargs["role"]
            if "verified" in kwargs:
                user.verified = kwargs["verified"]
            if "password" in kwargs:
                user.password_hash = bcrypt.hashpw(kwargs["password"].encode(), bcrypt.gensalt()).decode()

            return True

    def delete_user(self, username: str) -> bool:
        """Delete a user by username."""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return False
            session.delete(user)
            return True

    def verify_user(self, username: str) -> bool:
        """Mark a user as verified."""
        return self.update_user(username, verified=True)

    # ─────────────────────────────────────────────────────
    # MCP Server management (per-user)
    # ─────────────────────────────────────────────────────

    def get_user_servers(self, user_id: str) -> List[dict]:
        """Get all MCP servers for a user."""
        with self.get_session() as session:
            servers = session.query(UserMCPServer).filter(UserMCPServer.user_id == user_id).all()
            return [
                {
                    "name": s.name,
                    "description": s.description,
                    "url": s.url,
                    "transport": s.transport,
                }
                for s in servers
            ]

    def add_user_server(self, user_id: str, name: str, description: str, url: str, transport: Optional[str] = None) -> dict:
        """Add an MCP server for a user. Raises ValueError if name already exists for this user."""
        with self.get_session() as session:
            existing = session.query(UserMCPServer).filter(
                UserMCPServer.user_id == user_id, UserMCPServer.name == name
            ).first()
            if existing:
                raise ValueError(f"Server '{name}' already exists.")

            server = UserMCPServer(
                user_id=user_id,
                name=name,
                description=description,
                url=url,
                transport=transport,
            )
            session.add(server)
            session.flush()
            return {"name": name, "description": description, "url": url, "transport": transport}

    def update_user_server(self, user_id: str, name: str, description: Optional[str] = None, url: Optional[str] = None, transport: Optional[str] = None) -> bool:
        """Update an MCP server for a user."""
        with self.get_session() as session:
            server = session.query(UserMCPServer).filter(
                UserMCPServer.user_id == user_id, UserMCPServer.name == name
            ).first()
            if not server:
                return False
            if description is not None:
                server.description = description
            if url is not None:
                server.url = url
            if transport is not None:
                server.transport = transport
            return True

    def delete_user_server(self, user_id: str, name: str) -> bool:
        """Delete an MCP server for a user."""
        with self.get_session() as session:
            server = session.query(UserMCPServer).filter(
                UserMCPServer.user_id == user_id, UserMCPServer.name == name
            ).first()
            if not server:
                return False
            session.delete(server)
            return True

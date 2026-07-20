"""
Mosaic Authentication System

Architecture designed for easy migration:
─────────────────────────────────────────
This module uses a UserProvider interface. Currently backed by environment
variables (EnvUserProvider). To migrate to a database:

  1. Create a class that implements UserProvider (get_user, verify_password)
  2. Swap `user_provider = EnvUserProvider()` → `user_provider = DBUserProvider()`
  3. Everything else (JWT, dependencies, endpoints) stays the same.

Security features:
- bcrypt password hashing (timing-safe, salted)
- JWT with expiry and issued-at claims
- Rate limiting on login (in-memory, per-IP)
- Fails closed: missing JWT_SECRET in production = startup error
- No secrets in source code
"""

import os
import time
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict

import bcrypt
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger("auth")


# =============================================================================
# Configuration
# =============================================================================

# In production, JWT_SECRET MUST be set explicitly. In dev, we auto-generate.
_env = os.getenv("ENVIRONMENT", "development")
JWT_SECRET = os.getenv("JWT_SECRET")

if not JWT_SECRET:
    if _env == "production":
        raise RuntimeError(
            "JWT_SECRET must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    JWT_SECRET = secrets.token_hex(32)
    logger.warning("JWT_SECRET not set — using auto-generated key (tokens won't survive restarts)")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


# =============================================================================
# Models
# =============================================================================

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    username: str
    role: str


class TokenUser(BaseModel):
    """Represents the authenticated user extracted from a JWT."""
    username: str
    role: str  # "admin" or "user"


# =============================================================================
# User Provider Interface (swap this to migrate to a database)
# =============================================================================

class UserProvider(ABC):
    """
    Abstract interface for user storage.
    
    To migrate to a database:
    1. Implement this interface with DB queries
    2. Replace `user_provider = EnvUserProvider()` below
    """

    @abstractmethod
    def get_user(self, username: str) -> Optional[Dict]:
        """
        Get user by username.
        Returns: {"username": str, "password_hash": str, "role": str} or None
        """
        pass

    @abstractmethod
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        pass


class EnvUserProvider(UserProvider):
    """
    User provider backed by environment variables.
    Suitable for dev/small deployments. Passwords are bcrypt-hashed at startup.
    
    Required .env vars:
        ADMIN_USERNAME, ADMIN_PASSWORD
        USER_USERNAME, USER_PASSWORD
    """

    def __init__(self):
        self.users: Dict[str, Dict] = {}
        self._load_users()

    def _load_users(self):
        """Load users from environment variables and hash their passwords."""
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_pass = os.getenv("ADMIN_PASSWORD")
        if admin_pass:
            self.users[admin_user] = {
                "username": admin_user,
                "password_hash": bcrypt.hashpw(admin_pass.encode(), bcrypt.gensalt()).decode(),
                "role": "admin",
            }
            logger.info(f"Loaded admin user: {admin_user}")

        normal_user = os.getenv("USER_USERNAME", "user")
        normal_pass = os.getenv("USER_PASSWORD")
        if normal_pass:
            self.users[normal_user] = {
                "username": normal_user,
                "password_hash": bcrypt.hashpw(normal_pass.encode(), bcrypt.gensalt()).decode(),
                "role": "user",
            }
            logger.info(f"Loaded user: {normal_user}")

        if not self.users:
            logger.error(
                "No users configured! Set ADMIN_PASSWORD and USER_PASSWORD in .env. "
                "All login attempts will fail."
            )

    def get_user(self, username: str) -> Optional[Dict]:
        return self.users.get(username)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


# --- Active provider (change this line to swap implementations) ---
user_provider: UserProvider = EnvUserProvider()


# =============================================================================
# Token Operations
# =============================================================================

def create_access_token(username: str, role: str) -> str:
    """Create a short-lived access token."""
    payload = {
        "sub": username,
        "role": role,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
        "jti": secrets.token_hex(16),  # unique token ID
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def create_refresh_token(username: str, role: str) -> str:
    """Create a longer-lived refresh token."""
    payload = {
        "sub": username,
        "role": role,
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def verify_token(token: str, expected_type: str = "access") -> TokenUser:
    """Verify a JWT and return the user. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])

        # Verify token type
        if payload.get("type") != expected_type:
            raise HTTPException(status_code=401, detail="Invalid token type.")

        return TokenUser(username=payload["sub"], role=payload["role"])

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user. Returns user dict on success, None on failure.
    Uses constant-time comparison via bcrypt.
    """
    user = user_provider.get_user(username)
    if not user:
        # Still run bcrypt to prevent timing attacks revealing valid usernames
        bcrypt.checkpw(password.encode(), bcrypt.hashpw(b"dummy", bcrypt.gensalt()))
        return None

    if not user_provider.verify_password(password, user["password_hash"]):
        return None

    return {"username": user["username"], "role": user["role"]}


# =============================================================================
# FastAPI Dependencies
# =============================================================================

security = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenUser:
    """
    FastAPI dependency: validates the Bearer token from the Authorization header.
    Returns the authenticated TokenUser.
    """
    return verify_token(credentials.credentials, expected_type="access")


async def require_admin(user: TokenUser = Depends(get_current_user)) -> TokenUser:
    """
    FastAPI dependency: requires admin role.
    Use as: dependencies=[Depends(require_admin)]
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user

"""
Security Module
---------------
JWT authentication, API key validation, and authorization utilities
for the FastAPI backend.
"""

import os
import time
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    APIKeyHeader,
)
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)

# ── Password Hashing ─────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Security Schemes ─────────────────────────────────────────────
http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(
    name=settings.api_key_header_name, auto_error=False
)


# ── Pydantic Models ──────────────────────────────────────────────
class TokenData(BaseModel):
    """JWT token payload data."""
    user_id: str
    username: str
    role: str = "user"
    exp: Optional[int] = None
    iat: Optional[int] = None


class TokenResponse(BaseModel):
    """Token pair response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInDB(BaseModel):
    """User record (simulated — production would use DynamoDB/Cognito)."""
    user_id: str
    username: str
    email: str
    hashed_password: str
    role: str = "user"
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None


# ── Simulated User Store ─────────────────────────────────────────
# In production, this would be DynamoDB or AWS Cognito
_users_db: Dict[str, UserInDB] = {}


def _init_default_users() -> None:
    """Initialize default admin and demo users."""
    if not _users_db:
        admin_id = "user-admin-001"
        _users_db[admin_id] = UserInDB(
            user_id=admin_id,
            username="admin",
            email="admin@animequotegenerator.com",
            hashed_password=pwd_context.hash("admin123"),
            role="admin",
            is_active=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        demo_id = "user-demo-001"
        _users_db[demo_id] = UserInDB(
            user_id=demo_id,
            username="demo",
            email="demo@animequotegenerator.com",
            hashed_password=pwd_context.hash("demo123"),
            role="user",
            is_active=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


_init_default_users()


# ── JWT Token Operations ─────────────────────────────────────────

def create_access_token(user_id: str, username: str, role: str = "user") -> str:
    """
    Create a JWT access token.

    Args:
        user_id: Unique user identifier
        username: User's display name
        role: User role (user, admin)

    Returns:
        Encoded JWT access token string
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(user_id: str, username: str, role: str = "user") -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: Unique user identifier
        username: User's display name
        role: User role (user, admin)

    Returns:
        Encoded JWT refresh token string
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)

    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "refresh",
        "jti": secrets.token_hex(16),  # Unique token identifier
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with extracted claims

    Raises:
        HTTPException: If token is invalid, expired, or malformed
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )

        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        role: str = payload.get("role", "user")

        if user_id is None or username is None:
            raise credentials_exception

        return TokenData(
            user_id=user_id,
            username=username,
            role=role,
            exp=payload.get("exp"),
            iat=payload.get("iat"),
        )

    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise credentials_exception


# ── User Management ──────────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticate a user by username and password.

    Args:
        username: User's username
        password: Plain-text password to verify

    Returns:
        UserInDB if authentication succeeds, None otherwise
    """
    for user in _users_db.values():
        if user.username == username and user.is_active:
            if pwd_context.verify(password, user.hashed_password):
                user.last_login = datetime.now(timezone.utc).isoformat()
                return user
    return None


def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    """Retrieve a user by their ID."""
    return _users_db.get(user_id)


def register_user(username: str, email: str, password: str, role: str = "user") -> UserInDB:
    """
    Register a new user.

    Args:
        username: Desired username
        email: User's email address
        password: Plain-text password (will be hashed)
        role: User role (default: "user")

    Returns:
        Created UserInDB record

    Raises:
        HTTPException: If username or email already exists
    """
    # Check for existing username
    for existing in _users_db.values():
        if existing.username == username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Username '{username}' already exists",
            )
        if existing.email == email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Email '{email}' already registered",
            )

    user_id = f"user-{secrets.token_hex(6)}"
    user = UserInDB(
        user_id=user_id,
        username=username,
        email=email,
        hashed_password=pwd_context.hash(password),
        role=role,
        is_active=True,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _users_db[user_id] = user
    logger.info(f"Registered new user: {username} ({user_id})")
    return user


# ── FastAPI Dependencies ─────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> UserInDB:
    """
    FastAPI dependency: Extract and validate JWT from Authorization header.

    Returns:
        Authenticated UserInDB

    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_token(credentials.credentials)

    # Verify token type is access (not refresh)
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type. Use access token.",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    user = get_user_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> Optional[UserInDB]:
    """
    FastAPI dependency: Optionally extract JWT user (no error if missing).
    Used for endpoints that work for both authenticated and anonymous users.
    """
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def require_admin(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """
    FastAPI dependency: Require the authenticated user to have admin role.

    Raises:
        HTTPException: 403 if user is not an admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def validate_api_key(
    api_key: str = Security(api_key_header),
) -> str:
    """
    FastAPI dependency: Validate API key from request header.

    Returns:
        The validated API key string

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required",
        )

    if api_key not in settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

    return api_key

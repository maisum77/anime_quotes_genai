"""
Authentication Endpoints
------------------------
API routes for user registration, login, and token management.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Depends

from ....core.security import (
    authenticate_user,
    register_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    UserInDB,
)
from ....core.config import settings
from ....models.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    RefreshTokenRequest,
    UserInfoResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate with username and password to receive JWT tokens.",
    responses={
        200: {"description": "Authentication successful"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
    },
)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate a user and return JWT access/refresh tokens.

    The access token should be included in the Authorization header
    as a Bearer token for all authenticated requests.
    """
    user = authenticate_user(request.username, request.password)

    if not user:
        logger.warning(f"Failed login attempt for username: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
    )
    refresh_token = create_refresh_token(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
    )

    logger.info(f"User logged in: {user.username} ({user.user_id})")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post(
    "/register",
    response_model=UserInfoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account.",
    responses={
        201: {"description": "User created successfully"},
        409: {"model": ErrorResponse, "description": "Username or email already exists"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def register(request: RegisterRequest) -> UserInfoResponse:
    """
    Register a new user account.

    Password must be at least 8 characters with at least one
    uppercase letter, one lowercase letter, and one digit.
    """
    user = register_user(
        username=request.username,
        email=request.email,
        password=request.password,
    )

    return UserInfoResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Exchange a valid refresh token for a new token pair.",
    responses={
        200: {"description": "Tokens refreshed"},
        401: {"model": ErrorResponse, "description": "Invalid or expired refresh token"},
    },
)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """
    Refresh authentication tokens.

    Provide a valid refresh token to receive a new access/refresh pair.
    The old refresh token is invalidated.
    """
    try:
        token_data = decode_token(request.refresh_token)

        # Verify it's a refresh token
        from jose import jwt as jose_jwt
        payload = jose_jwt.decode(
            request.refresh_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type. Expected refresh token.",
            )

        # Issue new token pair
        access_token = create_access_token(
            user_id=token_data.user_id,
            username=token_data.username,
            role=token_data.role,
        )
        new_refresh = create_refresh_token(
            user_id=token_data.user_id,
            username=token_data.username,
            role=token_data.role,
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh,
            token_type="bearer",
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


@router.get(
    "/me",
    response_model=UserInfoResponse,
    summary="Get current user profile",
    description="Get the authenticated user's profile information.",
    responses={
        200: {"description": "User profile"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
    },
)
async def get_current_user_profile(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInfoResponse:
    """Get the authenticated user's profile information."""
    return UserInfoResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
    )

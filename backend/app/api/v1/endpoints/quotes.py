"""
Quotes Endpoints
----------------
API routes for retrieving generated anime quotes.
Supports listing, individual retrieval, and random quote access.
"""

import logging
import random
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....core.security import (
    get_current_user_optional,
    validate_api_key,
    UserInDB,
)
from ....core.config import settings
from ....models.schemas import (
    QuoteResponse,
    QuoteListResponse,
    RandomQuoteResponse,
    ErrorResponse,
)
from ....services.aws_client import api_gateway_client, get_api_gateway_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quotes", tags=["Quotes"])

# Fallback quotes for when API Gateway is not configured
FALLBACK_QUOTES = [
    {
        "id": "fallback-001",
        "content": "Listen everyone! True strength comes from within — never give up on your dreams!",
        "speech_type": "motivational",
        "generation_type": "speech",
        "generation_method": "fallback",
    },
    {
        "id": "fallback-002",
        "content": "This battle isn't over! My power has no limits when I fight for my friends!",
        "speech_type": "battle",
        "generation_type": "speech",
        "generation_method": "fallback",
    },
    {
        "id": "fallback-003",
        "content": "We're not alone — together our bonds make us unstoppable!",
        "speech_type": "friendship",
        "generation_type": "speech",
        "generation_method": "fallback",
    },
    {
        "id": "fallback-004",
        "content": "I made a promise I intend to keep, no matter the cost!",
        "speech_type": "determination",
        "generation_type": "speech",
        "generation_method": "fallback",
    },
    {
        "id": "fallback-005",
        "content": "You fools cling to hope while the world crumbles beneath your feet!",
        "speech_type": "villain",
        "generation_type": "speech",
        "generation_method": "fallback",
    },
]


@router.get(
    "",
    response_model=QuoteListResponse,
    summary="List generated quotes",
    description="Retrieve a paginated list of generated quotes. Requires API key.",
    responses={
        200: {"description": "Paginated list of quotes"},
        401: {"model": ErrorResponse, "description": "API key required"},
    },
)
async def list_quotes(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    speech_type: Optional[str] = Query(
        default=None, description="Filter by speech type"
    ),
    api_key: str = Depends(validate_api_key),
    current_user: Optional[UserInDB] = Depends(get_current_user_optional),
    client=Depends(get_api_gateway_client),
) -> QuoteListResponse:
    """
    List generated quotes with pagination and optional filtering.

    Requires a valid API key in the X-API-Key header.
    Supports filtering by speech_type.
    """
    logger.info(
        f"List quotes: page={page}, page_size={page_size}, "
        f"speech_type={speech_type}, user={current_user.user_id if current_user else 'api_key'}"
    )

    if not client.is_configured:
        # Fallback: return fallback quotes
        quotes = FALLBACK_QUOTES
        if speech_type:
            quotes = [q for q in quotes if q["speech_type"] == speech_type]

        start = (page - 1) * page_size
        end = start + page_size
        page_quotes = quotes[start:end]

        return QuoteListResponse(
            quotes=[
                QuoteResponse(
                    id=q["id"],
                    content=q["content"],
                    speech_type=q["speech_type"],
                    generation_type=q["generation_type"],
                    generation_method=q["generation_method"],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                for q in page_quotes
            ],
            total=len(quotes),
            page=page,
            page_size=page_size,
            has_next=end < len(quotes),
        )

    try:
        result = await client.get_quotes(
            page=page,
            page_size=page_size,
            speech_type=speech_type,
            api_key=api_key,
        )

        quotes = [
            QuoteResponse(
                id=q.get("id", ""),
                content=q.get("content", ""),
                speech_type=q.get("speech_type", ""),
                generation_type=q.get("generation_type", ""),
                generation_method=q.get("generation_method", ""),
                characters=q.get("characters", []),
                metadata=q.get("metadata", {}),
                created_at=q.get("created_at"),
            )
            for q in result.get("quotes", [])
        ]

        return QuoteListResponse(
            quotes=quotes,
            total=result.get("total", 0),
            page=result.get("page", page),
            page_size=result.get("page_size", page_size),
            has_next=result.get("has_next", False),
        )

    except Exception as e:
        logger.error(f"Failed to list quotes: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve quotes: {str(e)}",
        )


@router.get(
    "/random",
    response_model=RandomQuoteResponse,
    summary="Get random quote",
    description="Get a random anime quote. Public endpoint — no authentication required.",
    responses={
        200: {"description": "Random anime quote"},
    },
)
async def get_random_quote(
    client=Depends(get_api_gateway_client),
) -> RandomQuoteResponse:
    """
    Get a random anime quote.

    This is a public endpoint that doesn't require authentication.
    Returns a randomly selected quote from the generated collection.
    """
    logger.debug("Random quote request")

    if not client.is_configured:
        # Fallback: return a random fallback quote
        quote = random.choice(FALLBACK_QUOTES)
        return RandomQuoteResponse(
            content=quote["content"],
            speech_type=quote["speech_type"],
            generation_method=quote["generation_method"],
        )

    try:
        result = await client.get_random_quote()
        return RandomQuoteResponse(
            content=result.get("content", ""),
            speech_type=result.get("speech_type", ""),
            generation_method=result.get("generation_method", "fallback"),
        )

    except Exception as e:
        logger.error(f"Failed to get random quote: {e}")
        # Graceful fallback
        quote = random.choice(FALLBACK_QUOTES)
        return RandomQuoteResponse(
            content=quote["content"],
            speech_type=quote["speech_type"],
            generation_method="fallback",
        )


@router.get(
    "/{quote_id}",
    response_model=QuoteResponse,
    summary="Get specific quote",
    description="Retrieve a specific quote by its ID. Requires API key.",
    responses={
        200: {"description": "Quote details"},
        401: {"model": ErrorResponse, "description": "API key required"},
        404: {"model": ErrorResponse, "description": "Quote not found"},
    },
)
async def get_quote(
    quote_id: str,
    api_key: str = Depends(validate_api_key),
    client=Depends(get_api_gateway_client),
) -> QuoteResponse:
    """
    Get a specific quote by its ID.

    Requires a valid API key in the X-API-Key header.
    """
    logger.debug(f"Get quote: {quote_id}")

    if not client.is_configured:
        # Fallback: search in fallback quotes
        for q in FALLBACK_QUOTES:
            if q["id"] == quote_id:
                return QuoteResponse(
                    id=q["id"],
                    content=q["content"],
                    speech_type=q["speech_type"],
                    generation_type=q["generation_type"],
                    generation_method=q["generation_method"],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quote '{quote_id}' not found",
        )

    try:
        result = await client.get_quote(quote_id=quote_id, api_key=api_key)
        return QuoteResponse(
            id=result.get("id", quote_id),
            content=result.get("content", ""),
            speech_type=result.get("speech_type", ""),
            generation_type=result.get("generation_type", ""),
            generation_method=result.get("generation_method", ""),
            characters=result.get("characters", []),
            metadata=result.get("metadata", {}),
            created_at=result.get("created_at"),
        )

    except Exception as e:
        logger.error(f"Failed to get quote {quote_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve quote: {str(e)}",
        )

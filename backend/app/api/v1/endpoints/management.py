"""
Management Endpoints
--------------------
Health check, statistics, and configuration management routes.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ....core.security import get_current_user, require_admin, UserInDB
from ....core.config import settings
from ....models.schemas import (
    HealthResponse,
    PipelineStatsResponse,
    ConfigurationResponse,
    ErrorResponse,
)
from ....services.aws_client import api_gateway_client, get_api_gateway_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Management"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the API and its dependencies.",
    responses={200: {"description": "Service health status"}},
)
async def health_check(
    client=Depends(get_api_gateway_client),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the API and checks connectivity
    to downstream services (API Gateway, etc.).
    """
    services: Dict[str, str] = {
        "fastapi": "healthy",
    }

    # Check API Gateway connectivity
    if client.is_configured:
        try:
            await client.get_health()
            services["api_gateway"] = "healthy"
        except Exception:
            services["api_gateway"] = "unhealthy"
    else:
        services["api_gateway"] = "not_configured"

    overall_status = "healthy" if all(
        v in ("healthy", "not_configured") for v in services.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
    )


@router.get(
    "/stats",
    response_model=PipelineStatsResponse,
    summary="Pipeline statistics",
    description="Get statistics about the generation pipeline. Requires authentication.",
    responses={
        200: {"description": "Pipeline statistics"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
    },
)
async def get_pipeline_stats(
    current_user: UserInDB = Depends(get_current_user),
    client=Depends(get_api_gateway_client),
) -> PipelineStatsResponse:
    """
    Get pipeline statistics.

    Returns aggregate metrics about generation jobs including
    success rates, processing times, and method distribution.
    """
    logger.info(f"Stats requested by user {current_user.user_id}")

    if not client.is_configured:
        # Return simulated stats
        return PipelineStatsResponse(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            pending_jobs=0,
            generation_method_counts={
                "gemini": 0,
                "gpt2": 0,
                "fallback": 0,
            },
            speech_type_counts={
                "motivational": 0,
                "battle": 0,
                "friendship": 0,
                "determination": 0,
                "villain": 0,
            },
        )

    try:
        result = await client.get_stats(user_id=current_user.user_id)

        return PipelineStatsResponse(
            total_jobs=result.get("total_jobs", 0),
            completed_jobs=result.get("completed_jobs", 0),
            failed_jobs=result.get("failed_jobs", 0),
            pending_jobs=result.get("pending_jobs", 0),
            average_processing_time=result.get("average_processing_time"),
            generation_method_counts=result.get("generation_method_counts", {}),
            speech_type_counts=result.get("speech_type_counts", {}),
            recent_error_rate=result.get("recent_error_rate"),
        )

    except Exception as e:
        logger.error(f"Failed to get pipeline stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve pipeline statistics: {str(e)}",
        )


@router.get(
    "/config",
    response_model=ConfigurationResponse,
    summary="Pipeline configuration",
    description="Get the current pipeline configuration. Admin access required.",
    responses={
        200: {"description": "Pipeline configuration"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
    },
)
async def get_pipeline_config(
    admin_user: UserInDB = Depends(require_admin),
    client=Depends(get_api_gateway_client),
) -> ConfigurationResponse:
    """
    Get pipeline configuration (admin only).

    Returns the current configuration of AWS resources and
    pipeline settings.
    """
    logger.info(f"Config requested by admin {admin_user.user_id}")

    if not client.is_configured:
        return ConfigurationResponse(
            api_gateway_url="(not configured)",
            aws_region=settings.aws_region,
            s3_bucket=settings.s3_bucket_name or "(not configured)",
            dynamodb_jobs_table=settings.dynamodb_jobs_table,
            dynamodb_history_table=settings.dynamodb_history_table,
            sqs_generation_queue=settings.sqs_generation_queue_url or "(not configured)",
            sns_topics={
                "generation_complete": settings.sns_generation_complete_topic or "(not configured)",
                "generation_failed": settings.sns_generation_failed_topic or "(not configured)",
            },
            rate_limits={
                "per_minute": settings.rate_limit_per_minute,
                "burst": settings.rate_limit_burst,
            },
        )

    try:
        result = await client.get_config(user_id=admin_user.user_id)

        return ConfigurationResponse(
            api_gateway_url=result.get("api_gateway_url", settings.api_gateway_base_url),
            aws_region=result.get("aws_region", settings.aws_region),
            s3_bucket=result.get("s3_bucket", settings.s3_bucket_name),
            dynamodb_jobs_table=result.get("dynamodb_jobs_table", settings.dynamodb_jobs_table),
            dynamodb_history_table=result.get("dynamodb_history_table", settings.dynamodb_history_table),
            sqs_generation_queue=result.get("sqs_generation_queue", settings.sqs_generation_queue_url),
            sns_topics=result.get("sns_topics", {}),
            rate_limits=result.get("rate_limits", {
                "per_minute": settings.rate_limit_per_minute,
                "burst": settings.rate_limit_burst,
            }),
        )

    except Exception as e:
        logger.error(f"Failed to get pipeline config: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve configuration: {str(e)}",
        )

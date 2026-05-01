"""
Generation Endpoints
--------------------
API routes for submitting and tracking anime quote generation jobs.
Proxies requests to the AWS Lambda pipeline via API Gateway.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query

from ....core.security import get_current_user, UserInDB
from ....core.config import settings
from ....models.schemas import (
    GenerationRequest,
    BatchGenerationRequest,
    JobSubmittedResponse,
    BatchJobSubmittedResponse,
    JobStatusResponse,
    JobStatus,
    ErrorResponse,
)
from ....services.aws_client import api_gateway_client, get_api_gateway_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["Generation"])


@router.post(
    "",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate anime quote/speech",
    description="Submit a single generation request to the processing pipeline.",
    responses={
        202: {"description": "Job accepted and queued for processing"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "API Gateway unavailable"},
    },
)
async def generate_single(
    request: GenerationRequest,
    current_user: UserInDB = Depends(get_current_user),
    client=Depends(get_api_gateway_client),
) -> JobSubmittedResponse:
    """
    Submit a single anime quote/speech generation request.

    The request is forwarded to the AWS Lambda pipeline via API Gateway.
    Returns immediately with a job_id for tracking progress.
    """
    logger.info(
        f"Generation request from user {current_user.user_id}: "
        f"type={request.generation_type}, speech={request.speech_type}"
    )

    if not client.is_configured:
        # Fallback: return a simulated response when API Gateway is not configured
        import secrets
        job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
        return JobSubmittedResponse(
            job_id=job_id,
            status="queued",
            generation_type=request.generation_type.value,
            speech_type=request.speech_type.value,
            estimated_time_seconds=15,
            status_url=f"{settings.api_v1_prefix}/generate/{job_id}/status",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    try:
        request_data = request.model_dump()
        # Convert enums to string values
        request_data["generation_type"] = request.generation_type.value
        request_data["speech_type"] = request.speech_type.value

        result = await client.submit_generation(
            request_data=request_data,
            user_id=current_user.user_id,
        )

        return JobSubmittedResponse(
            job_id=result.get("job_id", "unknown"),
            status=result.get("status", "queued"),
            generation_type=request.generation_type.value,
            speech_type=request.speech_type.value,
            estimated_time_seconds=result.get("estimated_time_seconds", 15),
            status_url=f"{settings.api_v1_prefix}/generate/{result.get('job_id', 'unknown')}/status",
            created_at=result.get("created_at", datetime.now(timezone.utc).isoformat()),
        )

    except Exception as e:
        logger.error(f"API Gateway request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to submit generation request: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchJobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch generate anime quotes",
    description="Submit multiple generation requests in a single batch (max 10).",
    responses={
        202: {"description": "Batch accepted and jobs queued"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def generate_batch(
    request: BatchGenerationRequest,
    current_user: UserInDB = Depends(get_current_user),
    client=Depends(get_api_gateway_client),
) -> BatchJobSubmittedResponse:
    """
    Submit a batch of generation requests.

    Each request in the batch is processed independently.
    Returns a batch_id and individual job_ids for tracking.
    """
    import secrets

    logger.info(
        f"Batch generation request from user {current_user.user_id}: "
        f"{len(request.requests)} jobs"
    )

    if not client.is_configured:
        # Fallback: simulate batch response
        jobs = []
        for req in request.requests:
            job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
            jobs.append(JobSubmittedResponse(
                job_id=job_id,
                status="queued",
                generation_type=req.generation_type.value,
                speech_type=req.speech_type.value,
                estimated_time_seconds=15,
                status_url=f"{settings.api_v1_prefix}/generate/{job_id}/status",
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

        return BatchJobSubmittedResponse(
            batch_id=f"batch-{secrets.token_hex(8)}",
            jobs=jobs,
            total_jobs=len(jobs),
        )

    try:
        batch_data = {
            "requests": [
                {
                    **req.model_dump(),
                    "generation_type": req.generation_type.value,
                    "speech_type": req.speech_type.value,
                }
                for req in request.requests
            ]
        }

        result = await client.submit_batch_generation(
            batch_data=batch_data,
            user_id=current_user.user_id,
        )

        jobs = []
        for job_result in result.get("jobs", []):
            job_id = job_result.get("job_id", "unknown")
            jobs.append(JobSubmittedResponse(
                job_id=job_id,
                status=job_result.get("status", "queued"),
                generation_type=job_result.get("generation_type", "speech"),
                speech_type=job_result.get("speech_type", "motivational"),
                estimated_time_seconds=job_result.get("estimated_time_seconds", 15),
                status_url=f"{settings.api_v1_prefix}/generate/{job_id}/status",
                created_at=job_result.get("created_at", datetime.now(timezone.utc).isoformat()),
            ))

        return BatchJobSubmittedResponse(
            batch_id=result.get("batch_id", f"batch-{secrets.token_hex(8)}"),
            jobs=jobs,
            total_jobs=len(jobs),
        )

    except Exception as e:
        logger.error(f"Batch API Gateway request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to submit batch request: {str(e)}",
        )


@router.get(
    "/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Check the current status of a generation job.",
    responses={
        200: {"description": "Job status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_job_status(
    job_id: str,
    current_user: UserInDB = Depends(get_current_user),
    client=Depends(get_api_gateway_client),
) -> JobStatusResponse:
    """
    Get the current status of a generation job.

    Returns job status, progress information, and results (if completed).
    """
    logger.debug(f"Status check for job {job_id} by user {current_user.user_id}")

    if not client.is_configured:
        # Fallback: return simulated status
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            generation_type="speech",
            speech_type="motivational",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    try:
        result = await client.get_job_status(
            job_id=job_id,
            user_id=current_user.user_id,
        )

        return JobStatusResponse(
            job_id=result.get("job_id", job_id),
            status=JobStatus(result.get("status", "pending")),
            generation_type=result.get("generation_type"),
            speech_type=result.get("speech_type"),
            generation_method=result.get("generation_method"),
            created_at=result.get("created_at"),
            updated_at=result.get("updated_at"),
            completed_at=result.get("completed_at"),
            s3_key=result.get("s3_key"),
            result=result.get("result"),
            error=result.get("error"),
            processing_time=result.get("processing_time"),
        )

    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve job status: {str(e)}",
        )

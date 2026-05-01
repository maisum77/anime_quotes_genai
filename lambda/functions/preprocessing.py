"""
Preprocessing Lambda Function

Responsible for:
1. Input validation and sanitization
2. Request parsing and normalization
3. Job ID generation
4. Initial request logging
5. Forwarding validated requests to generation queue
"""
import json
import os
import time
import uuid
from typing import Dict, Any, Optional

from shared import (
    setup_logging, log_event, log_metric, log_error,
    validate_generation_request, validate_batch_request, validate_json_body,
    parse_api_gateway_event, ok_response, bad_request_response,
    internal_error_response, cors_preflight_response,
    accepted_response, not_found_response, unauthorized_response,
    send_to_sqs, generate_s3_key, upload_to_s3,
    ApiRouter
)

logger = setup_logging()

def generate_job_id() -> str:
    """
    Generate unique job ID
    
    Returns:
        Job ID string
    """
    return f"job-{uuid.uuid4().hex[:16]}-{int(time.time())}"

def preprocess_single_request(
    body: Dict[str, Any],
    request_id: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Preprocess single generation request
    
    Args:
        body: Request body
        request_id: Request ID
        user_id: Optional user ID
    
    Returns:
        Preprocessed request dictionary
    """
    start_time = time.time()
    
    try:
        # Validate request
        is_valid, error_message, validated_body = validate_generation_request(body)
        
        if not is_valid:
            log_metric("ValidationErrors", 1, dimensions={"ErrorType": "InvalidRequest"})
            raise ValueError(error_message)
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Create preprocessed request
        preprocessed_request = {
            "job_id": job_id,
            "request_id": request_id,
            "user_id": user_id,
            "request_type": "single",
            "validated_body": validated_body,
            "timestamp": int(time.time()),
            "status": "preprocessed",
            "processing_stage": "preprocessing"
        }
        
        # Log successful preprocessing
        logger.info(
            "Preprocessed single request | job_id=%s | speech_type=%s | generation_type=%s",
            job_id, validated_body.get("speech_type"), validated_body.get("generation_type")
        )
        
        log_metric("PreprocessingSuccess", 1, dimensions={"RequestType": "single"})
        
        # Log execution time
        execution_time = time.time() - start_time
        log_metric("PreprocessingTime", execution_time, unit="Seconds", dimensions={"RequestType": "single"})
        
        return preprocessed_request
        
    except Exception as e:
        log_error(e, {"request_id": request_id, "stage": "preprocessing"})
        log_metric("PreprocessingErrors", 1, dimensions={"ErrorType": type(e).__name__})
        raise

def preprocess_batch_request(
    body: Dict[str, Any],
    request_id: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Preprocess batch generation request
    
    Args:
        body: Request body
        request_id: Request ID
        user_id: Optional user ID
    
    Returns:
        Preprocessed batch request dictionary
    """
    start_time = time.time()
    
    try:
        # Validate batch request
        is_valid, error_message, validated_body = validate_batch_request(body)
        
        if not is_valid:
            log_metric("ValidationErrors", 1, dimensions={"ErrorType": "InvalidBatchRequest"})
            raise ValueError(error_message)
        
        # Generate batch job ID
        batch_job_id = generate_job_id()
        
        # Create individual job IDs for each item in batch
        count = validated_body["count"]
        speech_types = validated_body["speech_types"]
        
        individual_jobs = []
        for i in range(count):
            job_id = generate_job_id()
            speech_type = speech_types[i % len(speech_types)]
            
            individual_jobs.append({
                "job_id": job_id,
                "speech_type": speech_type,
                "generation_type": validated_body.get("generation_type", "speech"),
                "characters": validated_body.get("characters", ["Hero", "Rival"]),
                "temperature": validated_body.get("temperature", 0.8),
                "sequence": i,
                "total": count
            })
        
        # Create preprocessed batch request
        preprocessed_request = {
            "batch_job_id": batch_job_id,
            "request_id": request_id,
            "user_id": user_id,
            "request_type": "batch",
            "validated_body": validated_body,
            "individual_jobs": individual_jobs,
            "timestamp": int(time.time()),
            "status": "preprocessed",
            "processing_stage": "preprocessing"
        }
        
        # Log successful preprocessing
        logger.info(
            "Preprocessed batch request | batch_job_id=%s | count=%d | speech_types=%s",
            batch_job_id, count, speech_types
        )
        
        log_metric("PreprocessingSuccess", 1, dimensions={"RequestType": "batch"})
        
        # Log execution time
        execution_time = time.time() - start_time
        log_metric("PreprocessingTime", execution_time, unit="Seconds", dimensions={"RequestType": "batch"})
        
        return preprocessed_request
        
    except Exception as e:
        log_error(e, {"request_id": request_id, "stage": "batch_preprocessing"})
        log_metric("PreprocessingErrors", 1, dimensions={"ErrorType": type(e).__name__})
        raise

def store_request_metadata(
    preprocessed_request: Dict[str, Any],
    s3_bucket: Optional[str] = None
) -> str:
    """
    Store preprocessed request metadata in S3
    
    Args:
        preprocessed_request: Preprocessed request dictionary
        s3_bucket: S3 bucket name
    
    Returns:
        S3 object URL
    """
    try:
        job_id = preprocessed_request.get("job_id") or preprocessed_request.get("batch_job_id")
        
        # Generate S3 key
        s3_key = generate_s3_key(
            prefix="inputs",
            job_id=job_id,
            filename="request.json",
            timestamp=preprocessed_request.get("timestamp")
        )
        
        # Upload to S3
        s3_url = upload_to_s3(
            data=json.dumps(preprocessed_request, indent=2),
            key=s3_key,
            bucket=s3_bucket,
            content_type="application/json",
            metadata={
                "job_id": job_id,
                "request_type": preprocessed_request.get("request_type", "unknown"),
                "timestamp": str(preprocessed_request.get("timestamp"))
            }
        )
        
        logger.info("Stored request metadata in S3 | job_id=%s | s3_url=%s", job_id, s3_url)
        
        return s3_url
        
    except Exception as e:
        logger.error("Failed to store request metadata: %s", e)
        # Don't fail the whole process if S3 upload fails
        return ""

def forward_to_generation_queue(
    preprocessed_request: Dict[str, Any],
    queue_name: str = "generation-queue"
) -> str:
    """
    Forward preprocessed request to generation queue
    
    Args:
        preprocessed_request: Preprocessed request dictionary
        queue_name: SQS queue name
    
    Returns:
        SQS message ID
    """
    try:
        message_id = send_to_sqs(
            message=preprocessed_request,
            queue_name=queue_name
        )
        
        logger.info(
            "Forwarded to generation queue | job_id=%s | message_id=%s",
            preprocessed_request.get("job_id") or preprocessed_request.get("batch_job_id"),
            message_id
        )
        
        log_metric("QueueMessagesSent", 1, dimensions={"Queue": queue_name})
        
        return message_id
        
    except Exception as e:
        logger.error("Failed to forward to generation queue: %s", e)
        log_metric("QueueErrors", 1, dimensions={"Queue": queue_name, "ErrorType": type(e).__name__})
        raise

def handle_single_generation(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle single generation request from API Gateway
    
    Args:
        event: API Gateway event
        context: Lambda context
    
    Returns:
        API Gateway response
    """
    try:
        log_event(event, context)
        
        # Parse event
        parsed_event = parse_api_gateway_event(event)
        
        # Handle OPTIONS request (CORS preflight)
        if parsed_event["method"] == "OPTIONS":
            return cors_preflight_response()
        
        # Validate JSON body
        body_str = event.get("body", "")
        is_valid, error_message, body = validate_json_body(body_str)
        
        if not is_valid:
            return bad_request_response(error_message)
        
        # Extract user ID from headers (if available)
        user_id = parsed_event["headers"].get("x-user-id")
        
        # Generate request ID
        request_id = context.aws_request_id if context else str(uuid.uuid4())
        
        # Preprocess request
        preprocessed_request = preprocess_single_request(
            body=body,
            request_id=request_id,
            user_id=user_id
        )
        
        # Store request metadata in S3
        s3_bucket = os.environ.get("S3_BUCKET")
        s3_url = store_request_metadata(preprocessed_request, s3_bucket)
        
        # Forward to generation queue
        queue_name = os.environ.get("GENERATION_QUEUE", "generation-queue")
        message_id = forward_to_generation_queue(preprocessed_request, queue_name)
        
        # Prepare response
        job_id = preprocessed_request["job_id"]
        response_body = {
            "job_id": job_id,
            "status": "queued",
            "message": "Request queued for processing",
            "queue_message_id": message_id,
            "request_id": request_id,
            "estimated_wait_time": "10-30 seconds",
            "s3_metadata_url": s3_url,
            "timestamp": preprocessed_request["timestamp"]
        }
        
        # Add links to check status
        api_gateway_id = os.environ.get("API_GATEWAY_ID", "api")
        region = os.environ.get("AWS_REGION", "us-east-1")
        
        if api_gateway_id != "api":
            response_body["status_url"] = f"https://{api_gateway_id}.execute-api.{region}.amazonaws.com/prod/status/{job_id}"
        
        return ok_response(response_body)
        
    except ValueError as e:
        # Validation error
        return bad_request_response(str(e))
        
    except Exception as e:
        # Internal error
        log_error(e, {"stage": "preprocessing_handler"})
        return internal_error_response(
            message="Failed to preprocess request",
            error_id=context.aws_request_id if context else None
        )

def handle_batch_generation(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle batch generation request from API Gateway
    
    Args:
        event: API Gateway event
        context: Lambda context
    
    Returns:
        API Gateway response
    """
    try:
        log_event(event, context)
        
        # Parse event
        parsed_event = parse_api_gateway_event(event)
        
        # Handle OPTIONS request (CORS preflight)
        if parsed_event["method"] == "OPTIONS":
            return cors_preflight_response()
        
        # Validate JSON body
        body_str = event.get("body", "")
        is_valid, error_message, body = validate_json_body(body_str)
        
        if not is_valid:
            return bad_request_response(error_message)
        
        # Extract user ID from headers (if available)
        user_id = parsed_event["headers"].get("x-user-id")
        
        # Generate request ID
        request_id = context.aws_request_id if context else str(uuid.uuid4())
        
        # Preprocess batch request
        preprocessed_request = preprocess_batch_request(
            body=body,
            request_id=request_id,
            user_id=user_id
        )
        
        # Store request metadata in S3
        s3_bucket = os.environ.get("S3_BUCKET")
        s3_url = store_request_metadata(preprocessed_request, s3_bucket)
        
        # Forward to generation queue
        queue_name = os.environ.get("GENERATION_QUEUE", "generation-queue")
        message_id = forward_to_generation_queue(preprocessed_request, queue_name)
        
        # Prepare response
        batch_job_id = preprocessed_request["batch_job_id"]
        individual_jobs = preprocessed_request["individual_jobs"]
        
        response_body = {
            "batch_job_id": batch_job_id,
            "status": "queued",
            "message": "Batch request queued for processing",
            "queue_message_id": message_id,
            "request_id": request_id,
            "job_count": len(individual_jobs),
            "individual_jobs": [
                {
                    "job_id": job["job_id"],
                    "speech_type": job["speech_type"],
                    "sequence": job["sequence"]
                }
                for job in individual_jobs
            ],
            "estimated_wait_time": f"{len(individual_jobs) * 10}-{len(individual_jobs) * 30} seconds",
            "s3_metadata_url": s3_url,
            "timestamp": preprocessed_request["timestamp"]
        }
        
        return ok_response(response_body)
        
    except ValueError as e:
        # Validation error
        return bad_request_response(str(e))
        
    except Exception as e:
        # Internal error
        log_error(e, {"stage": "batch_preprocessing_handler"})
        return internal_error_response(
            message="Failed to preprocess batch request",
            error_id=context.aws_request_id if context else None
        )

# ---------------------------------------------------------------------------
# API Router Setup
# ---------------------------------------------------------------------------

router = ApiRouter()


def handle_health_check(request: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Health check endpoint handler.
    
    Returns service health status for all pipeline components.
    """
    health_status = {
        "status": "healthy",
        "version": os.environ.get("API_VERSION", "1.0.0"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": {
            "preprocessing": "healthy",
            "generation": "healthy",
            "postprocessing": "healthy",
            "s3": "healthy",
            "sqs": "healthy",
            "dynamodb": "healthy",
        }
    }
    
    # Check SQS connectivity
    try:
        queue_url = os.environ.get(ENV_GENERATION_QUEUE, "")
        if queue_url:
            import boto3
            sqs = boto3.client("sqs")
            sqs.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=["ApproximateNumberOfMessages"]
            )
    except Exception as e:
        health_status["services"]["sqs"] = f"degraded: {str(e)[:50]}"
        health_status["status"] = "degraded"
    
    return ok_response(health_status)


def handle_get_job_status(request: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Get generation job status by job_id.
    
    Looks up job metadata in DynamoDB and returns current status.
    """
    job_id = request.get("pathParameters", {}).get("job_id")
    
    if not job_id:
        return bad_request_response("Missing job_id parameter")
    
    try:
        import boto3
        dynamodb = boto3.resource("dynamodb")
        table_name = os.environ.get(ENV_JOBS_TABLE, DYNAMODB_JOBS_TABLE)
        table = dynamodb.Table(table_name)
        
        response = table.get_item(Key={"job_id": job_id})
        item = response.get("Item")
        
        if not item:
            return not_found_response(f"Job {job_id} not found", resource="job")
        
        # Build status response
        status_body = {
            "job_id": item.get("job_id"),
            "status": item.get("status", "unknown"),
            "generation_type": item.get("generation_type"),
            "speech_type": item.get("speech_type"),
            "result": item.get("result"),
            "timing": {
                "queued_at": item.get("queued_at"),
                "processing_started_at": item.get("processing_started_at"),
                "completed_at": item.get("completed_at"),
                "total_duration_seconds": item.get("total_duration_seconds"),
            },
            "metadata": {
                "s3_key": item.get("s3_key"),
                "request_id": item.get("request_id"),
            }
        }
        
        return ok_response(status_body)
        
    except Exception as e:
        log_error(e, {"job_id": job_id, "operation": "get_job_status"})
        return internal_error_response(
            message="Failed to retrieve job status",
            error_id=job_id
        )


def handle_get_random_quote(request: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Get a random anime quote (public endpoint, no auth required).
    
    Returns a random quote from the fallback prompts or S3 storage.
    """
    import random
    
    # Try to get from S3 first
    try:
        bucket = os.environ.get(ENV_S3_BUCKET, "")
        if bucket:
            import boto3
            s3 = boto3.client("s3")
            # List recent outputs
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{S3_OUTPUTS_PREFIX}/",
                MaxKeys=100
            )
            
            objects = response.get("Contents", [])
            if objects:
                # Pick a random object
                obj = random.choice(objects)
                obj_response = s3.get_object(Bucket=bucket, Key=obj["Key"])
                content = json.loads(obj_response["Body"].read().decode("utf-8"))
                
                return ok_response({
                    "quote": {
                        "content": content.get("content", ""),
                        "character": content.get("character", ""),
                        "anime": content.get("anime", "Unknown"),
                        "speech_type": content.get("speech_type", ""),
                    },
                    "metadata": {
                        "generated_at": content.get("created_at", ""),
                        "model_used": content.get("model_used", ""),
                    }
                })
    except Exception as e:
        logger.warning("Failed to get random quote from S3: %s", e)
    
    # Fallback to static prompts
    from shared.constants import FALLBACK_PROMPTS, DEFAULT_SPEECH_TYPE
    speech_type = random.choice(list(FALLBACK_PROMPTS.keys()))
    quote = random.choice(FALLBACK_PROMPTS[speech_type])
    
    return ok_response({
        "quote": {
            "content": quote,
            "character": "Anime Character",
            "anime": "Unknown",
            "speech_type": speech_type,
        },
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_used": "fallback",
        }
    })


def handle_get_stats(request: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Get pipeline statistics (requires JWT auth).
    
    Returns aggregate statistics about generation requests.
    """
    try:
        import boto3
        dynamodb = boto3.resource("dynamodb")
        table_name = os.environ.get(ENV_JOBS_TABLE, DYNAMODB_JOBS_TABLE)
        table = dynamodb.Table(table_name)
        
        # Scan for stats (in production, use a dedicated stats table)
        response = table.scan(Select="COUNT")
        total_jobs = response.get("Count", 0)
        
        stats = {
            "total_jobs": total_jobs,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        
        return ok_response(stats)
        
    except Exception as e:
        log_error(e, {"operation": "get_stats"})
        return internal_error_response("Failed to retrieve statistics")


def handle_single_route(request: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Route handler that adapts API router requests to handle_single_generation."""
    # Convert router request format back to API Gateway event format
    event = _request_to_event(request)
    return handle_single_generation(event, context)


def handle_batch_route(request: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Route handler that adapts API router requests to handle_batch_generation."""
    event = _request_to_event(request)
    return handle_batch_generation(event, context)


def _request_to_event(request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert API router parsed request back to API Gateway event format."""
    return {
        "httpMethod": request.get("http_method", "POST"),
        "path": request.get("path", "/"),
        "headers": request.get("headers", {}),
        "queryStringParameters": request.get("query_params", {}),
        "pathParameters": request.get("path_parameters", {}),
        "body": json.dumps(request.get("body", {})) if isinstance(request.get("body"), dict) else request.get("body", ""),
        "requestContext": {
            "requestId": request.get("request_id", ""),
            "authorizer": {
                "jwt": {"claims": request.get("authorizer_claims", {})}
            } if request.get("authorizer_claims") else {}
        }
    }


# Register API routes
router.add_route("POST", "/v1/generate", handle_single_route, auth_required=True, rate_limit=10)
router.add_route("POST", "/v1/generate/batch", handle_batch_route, auth_required=True, rate_limit=2)
router.add_route("GET", "/v1/generate/{job_id}/status", handle_get_job_status, auth_required=True)
router.add_route("GET", "/v1/health", handle_health_check)
router.add_route("GET", "/v1/quotes/random", handle_get_random_quote)
router.add_route("GET", "/v1/stats", handle_get_stats, auth_required=True)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for preprocessing function.
    
    Uses the API router for request dispatching. Falls back to
    direct path-based routing for backward compatibility.
    
    Args:
        event: Lambda event (API Gateway HTTP API v2 format)
        context: Lambda context
    
    Returns:
        API Gateway response
    """
    try:
        # Handle CORS preflight
        http_method = (
            event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod", "")
        )
        if http_method.upper() == "OPTIONS":
            return cors_preflight_response()
        
        # Try API router first (for HTTP API v2 events with routeKey)
        route_key = event.get("routeKey", "")
        raw_path = event.get("rawPath", "")
        
        # If this is an HTTP API v2 event with a recognizable path, use the router
        if raw_path and "/v1/" in raw_path:
            return router.dispatch(event, context)
        
        # Fallback: legacy path-based routing for direct invocations
        parsed_event = parse_api_gateway_event(event)
        path = parsed_event["path"]
        
        if "/batch" in path.lower():
            return handle_batch_generation(event, context)
        else:
            return handle_single_generation(event, context)
            
    except Exception as e:
        log_error(e, {"stage": "preprocessing_lambda_handler"})
        return internal_error_response(
            message="Internal server error in preprocessing",
            error_id=context.aws_request_id if context else None
        )


# For direct testing
if __name__ == "__main__":
    # Test the preprocessing function
    test_event = {
        "httpMethod": "POST",
        "path": "/generate",
        "headers": {
            "Content-Type": "application/json",
            "X-User-ID": "test-user-123"
        },
        "body": json.dumps({
            "speech_type": "motivational",
            "type": "speech",
            "temperature": 0.8
        })
    }
    
    class TestContext:
        aws_request_id = "test-request-123"
        function_name = "test-preprocessing"
    
    context = TestContext()
    
    result = lambda_handler(test_event, context)
    print("Test Result:", json.dumps(result, indent=2))
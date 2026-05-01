"""
HTTP utilities for API Gateway responses
"""
import json
from typing import Dict, Any, Optional
from .constants import (
    HTTP_OK, HTTP_BAD_REQUEST, HTTP_UNAUTHORIZED, HTTP_FORBIDDEN,
    HTTP_NOT_FOUND, HTTP_METHOD_NOT_ALLOWED, HTTP_TOO_MANY_REQUESTS,
    HTTP_INTERNAL_ERROR, HTTP_BAD_GATEWAY, HTTP_SERVICE_UNAVAILABLE,
    HTTP_GATEWAY_TIMEOUT
)

def cors_headers() -> Dict[str, str]:
    """
    Get CORS headers for API Gateway responses
    
    Returns:
        Dictionary of CORS headers
    """
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-API-Key",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        "Access-Control-Allow-Credentials": "true",
    }

def create_response(
    status_code: int,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create API Gateway response
    
    Args:
        status_code: HTTP status code
        body: Response body
        headers: Additional headers
    
    Returns:
        API Gateway response dictionary
    """
    response_headers = cors_headers()
    if headers:
        response_headers.update(headers)
    
    return {
        "statusCode": status_code,
        "headers": response_headers,
        "body": json.dumps(body, default=str)
    }

def ok_response(
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create 200 OK response
    
    Args:
        body: Response body
        headers: Additional headers
    
    Returns:
        API Gateway response dictionary
    """
    return create_response(HTTP_OK, body, headers)

def bad_request_response(
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create 400 Bad Request response
    
    Args:
        message: Error message
        details: Additional error details
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message}
    if details:
        body["details"] = details
    
    return create_response(HTTP_BAD_REQUEST, body)

def not_found_response(
    message: str,
    resource: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create 404 Not Found response
    
    Args:
        message: Error message
        resource: Resource that was not found
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message}
    if resource:
        body["resource"] = resource
    
    return create_response(HTTP_NOT_FOUND, body)

def internal_error_response(
    message: str = "Internal server error",
    error_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create 500 Internal Server Error response
    
    Args:
        message: Error message
        error_id: Optional error ID for tracking
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message}
    if error_id:
        body["error_id"] = error_id
    
    return create_response(HTTP_INTERNAL_ERROR, body)

def unauthorized_response(
    message: str = "Authentication required",
    code: str = "UNAUTHORIZED"
) -> Dict[str, Any]:
    """
    Create 401 Unauthorized response
    
    Args:
        message: Error message
        code: Error code
    
    Returns:
        API Gateway response dictionary
    """
    return create_response(HTTP_UNAUTHORIZED, {"error": message, "code": code})

def forbidden_response(
    message: str = "Access denied",
    code: str = "FORBIDDEN"
) -> Dict[str, Any]:
    """
    Create 403 Forbidden response
    
    Args:
        message: Error message
        code: Error code
    
    Returns:
        API Gateway response dictionary
    """
    return create_response(HTTP_FORBIDDEN, {"error": message, "code": code})

def method_not_allowed_response(
    message: str = "Method not allowed",
    allowed_methods: Optional[list] = None
) -> Dict[str, Any]:
    """
    Create 405 Method Not Allowed response
    
    Args:
        message: Error message
        allowed_methods: List of allowed HTTP methods
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message, "code": "METHOD_NOT_ALLOWED"}
    if allowed_methods:
        body["allowed_methods"] = allowed_methods
    return create_response(HTTP_METHOD_NOT_ALLOWED, body)

def rate_limited_response(
    retry_after: int = 30,
    limit: int = 0,
    remaining: int = 0,
    reset_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create 429 Too Many Requests response
    
    Args:
        retry_after: Seconds until client should retry
        limit: Rate limit ceiling
        remaining: Remaining requests in window
        reset_at: ISO timestamp when the limit resets
    
    Returns:
        API Gateway response dictionary
    """
    body = {
        "error": "Rate limit exceeded",
        "code": "RATE_LIMITED",
        "retry_after_seconds": retry_after,
        "limit": limit,
        "remaining": remaining
    }
    if reset_at:
        body["reset_at"] = reset_at
    
    headers = {"Retry-After": str(retry_after)}
    return create_response(HTTP_TOO_MANY_REQUESTS, body, headers)

def bad_gateway_response(
    message: str = "Upstream service error",
    service: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create 502 Bad Gateway response
    
    Args:
        message: Error message
        service: Name of the upstream service that failed
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message, "code": "UPSTREAM_ERROR"}
    if service:
        body["service"] = service
    return create_response(HTTP_BAD_GATEWAY, body)

def service_unavailable_response(
    message: str = "Service temporarily unavailable",
    retry_after: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create 503 Service Unavailable response
    
    Args:
        message: Error message
        retry_after: Seconds until client should retry
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message, "code": "SERVICE_UNAVAILABLE"}
    headers = {}
    if retry_after:
        body["retry_after_seconds"] = retry_after
        headers["Retry-After"] = str(retry_after)
    return create_response(HTTP_SERVICE_UNAVAILABLE, body, headers)

def timeout_response(
    message: str = "Request processing timeout",
    timeout_seconds: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create 504 Gateway Timeout response
    
    Args:
        message: Error message
        timeout_seconds: Timeout duration in seconds
    
    Returns:
        API Gateway response dictionary
    """
    body = {"error": message, "code": "TIMEOUT"}
    if timeout_seconds:
        body["timeout_seconds"] = timeout_seconds
    return create_response(HTTP_GATEWAY_TIMEOUT, body)

def accepted_response(
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create 202 Accepted response (for async processing)
    
    Args:
        body: Response body
        headers: Additional headers
    
    Returns:
        API Gateway response dictionary
    """
    return create_response(202, body, headers)

def cors_preflight_response() -> Dict[str, Any]:
    """
    Create CORS preflight response for OPTIONS requests
    
    Returns:
        API Gateway response dictionary
    """
    return {
        "statusCode": 200,
        "headers": cors_headers(),
        "body": ""
    }

def parse_api_gateway_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse API Gateway event and extract common fields
    
    Args:
        event: API Gateway event
    
    Returns:
        Dictionary with parsed event data
    """
    # Handle both REST API and HTTP API formats
    request_context = event.get("requestContext", {})
    
    # HTTP API v2 format
    if "http" in request_context:
        http_info = request_context["http"]
        method = http_info.get("method", "GET")
        path = event.get("rawPath", "/")
        source_ip = http_info.get("sourceIp", "")
        user_agent = http_info.get("userAgent", "")
    # REST API format
    else:
        method = event.get("httpMethod", "GET")
        path = event.get("path", "/")
        source_ip = request_context.get("identity", {}).get("sourceIp", "")
        user_agent = request_context.get("identity", {}).get("userAgent", "")
    
    # Parse query string parameters
    query_params = event.get("queryStringParameters", {}) or {}
    
    # Parse path parameters
    path_params = event.get("pathParameters", {}) or {}
    
    # Parse headers (case-insensitive)
    headers = {k.lower(): v for k, v in (event.get("headers", {}) or {}).items()}
    
    # Parse body
    body = event.get("body", "")
    if body and headers.get("content-type", "").startswith("application/json"):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {"_raw": body}
    
    return {
        "method": method.upper(),
        "path": path,
        "source_ip": source_ip,
        "user_agent": user_agent,
        "query_params": query_params,
        "path_params": path_params,
        "headers": headers,
        "body": body,
        "raw_event": event
    }

def extract_api_key(event: Dict[str, Any]) -> Optional[str]:
    """
    Extract API key from event headers
    
    Args:
        event: API Gateway event
    
    Returns:
        API key or None
    """
    headers = event.get("headers", {}) or {}
    
    # Check various header names for API key
    api_key_headers = ["x-api-key", "api-key", "authorization"]
    
    for header_name in api_key_headers:
        if header_name in headers:
            return headers[header_name]
    
    return None

def validate_api_key(api_key: str, valid_keys: Optional[list] = None) -> bool:
    """
    Validate API key
    
    Args:
        api_key: API key to validate
        valid_keys: List of valid API keys (from environment if None)
    
    Returns:
        True if valid, False otherwise
    """
    import os
    
    if valid_keys is None:
        # Get valid keys from environment
        env_keys = os.environ.get("API_KEYS", "")
        valid_keys = [key.strip() for key in env_keys.split(",") if key.strip()]
    
    return api_key in valid_keys

def rate_limit_check(
    client_ip: str,
    endpoint: str,
    max_requests: int = 100,
    window_seconds: int = 3600
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Simple rate limiting check (in production, use API Gateway rate limiting)
    
    Args:
        client_ip: Client IP address
        endpoint: API endpoint
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
    
    Returns:
        Tuple of (is_allowed, rate_limit_info)
    """
    # In production, this would use DynamoDB or Redis for rate limiting
    # This is a simplified implementation
    import time
    
    # Mock implementation - always allow for now
    # In real implementation, you would:
    # 1. Store request counts in DynamoDB with TTL
    # 2. Check current count
    # 3. Increment if below limit
    
    current_time = int(time.time())
    window_start = current_time - window_seconds
    
    # For now, always allow
    is_allowed = True
    
    rate_limit_info = {
        "limit": max_requests,
        "remaining": max_requests - 1,  # Mock value
        "reset": window_start + window_seconds,
        "window_seconds": window_seconds
    }
    
    return is_allowed, rate_limit_info
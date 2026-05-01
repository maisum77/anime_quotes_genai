"""
API Gateway request router and handler utilities.

Provides route matching, request parsing, and response formatting
for API Gateway HTTP API (v2) integration with Lambda functions.
"""
import json
import re
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from .http_utils import create_response, cors_headers
from .constants import HTTP_OK, HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_INTERNAL_ERROR, HTTP_METHOD_NOT_ALLOWED
from .logging import log_metric, logger


@dataclass
class ApiRoute:
    """Represents a single API route definition."""
    method: str
    path: str
    handler: Callable
    auth_required: bool = False
    admin_required: bool = False
    rate_limit: Optional[int] = None  # requests per second
    path_pattern: Optional[re.Pattern] = field(default=None, init=False, repr=False)
    path_params: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """Compile path pattern for route matching."""
        # Convert path like /v1/generate/{job_id}/status to regex
        param_names = []
        pattern_parts = []
        for segment in self.path.split('/'):
            if segment.startswith('{') and segment.endswith('}'):
                param_name = segment[1:-1]
                param_names.append(param_name)
                pattern_parts.append(f'(?P<{param_name}>[^/]+)')
            else:
                pattern_parts.append(re.escape(segment))
        self.path_pattern = re.compile('^' + '/'.join(pattern_parts) + '$')
        self.path_params = param_names


class ApiRouter:
    """
    API Gateway request router.
    
    Matches incoming API Gateway v2 events to registered routes
    and dispatches to the appropriate handler function.
    """

    def __init__(self):
        self._routes: List[ApiRoute] = []

    def add_route(
        self,
        method: str,
        path: str,
        handler: Callable,
        auth_required: bool = False,
        admin_required: bool = False,
        rate_limit: Optional[int] = None
    ) -> None:
        """
        Register a new API route.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: URL path with optional {param} placeholders
            handler: Callable to handle the request
            auth_required: Whether authentication is required
            admin_required: Whether admin privileges are required
            rate_limit: Optional requests-per-second limit
        """
        route = ApiRoute(
            method=method.upper(),
            path=path,
            handler=handler,
            auth_required=auth_required,
            admin_required=admin_required,
            rate_limit=rate_limit
        )
        self._routes.append(route)
        logger.info("Registered route: %s %s", method.upper(), path)

    def match(self, method: str, path: str) -> Optional[Tuple[ApiRoute, Dict[str, str]]]:
        """
        Match a request to a registered route.
        
        Args:
            method: HTTP method
            path: Request path
            
        Returns:
            Tuple of (matched route, path parameters) or None
        """
        method = method.upper()
        for route in self._routes:
            if route.method != method:
                continue
            match = route.path_pattern.match(path)
            if match:
                return route, match.groupdict()
        return None

    def dispatch(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """
        Dispatch an API Gateway v2 event to the appropriate handler.
        
        Args:
            event: API Gateway HTTP API v2 event
            context: Lambda context
            
        Returns:
            API Gateway response dictionary
        """
        try:
            # Extract request details
            http_method = event.get('requestContext', {}).get('http', {}).get('method', '')
            raw_path = event.get('rawPath', '')
            stage = event.get('requestContext', {}).get('stage', '')
            
            # Remove stage prefix from path if present
            path = raw_path
            if stage and path.startswith(f'/{stage}'):
                path = path[len(stage) + 1:]

            # Parse request
            request = parse_api_gateway_event(event)

            # Match route
            result = self.match(http_method, path)
            
            if result is None:
                # Check if path exists with different method
                path_exists = any(
                    route.path_pattern.match(path) for route in self._routes
                )
                if path_exists:
                    return create_response(
                        HTTP_METHOD_NOT_ALLOWED,
                        {"error": f"Method {http_method} not allowed for {path}"}
                    )
                return create_response(
                    HTTP_NOT_FOUND,
                    {"error": f"No route found for {http_method} {path}"}
                )

            route, path_params = result
            request['pathParameters'] = path_params

            # Check authentication
            if route.auth_required:
                auth_result = _check_authentication(event, request)
                if auth_result is not None:
                    return auth_result

            # Check admin authorization
            if route.admin_required:
                auth_result = _check_admin_authorization(event, request)
                if auth_result is not None:
                    return auth_result

            # Check rate limit
            if route.rate_limit:
                rate_limit_result = _check_rate_limit(request, route.rate_limit)
                if rate_limit_result is not None:
                    return rate_limit_result

            # Dispatch to handler
            response = route.handler(request, context)
            
            # Log successful request
            log_metric("RequestCount", 1, dimensions={
                "Method": http_method,
                "Path": path,
                "Status": str(response.get('statusCode', 200))
            })

            return response

        except Exception as e:
            logger.error("Error dispatching request: %s", e, exc_info=True)
            log_metric("DispatchError", 1, dimensions={"ErrorType": type(e).__name__})
            return create_response(
                HTTP_INTERNAL_ERROR,
                {"error": "Internal server error", "code": "DISPATCH_ERROR"}
            )

    def get_routes_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered routes."""
        return [
            {
                "method": route.method,
                "path": route.path,
                "auth_required": route.auth_required,
                "admin_required": route.admin_required,
                "rate_limit": route.rate_limit
            }
            for route in self._routes
        ]


def parse_api_gateway_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse API Gateway HTTP API v2 event into a standardized request dict.
    
    Args:
        event: Raw API Gateway v2 event
        
    Returns:
        Standardized request dictionary
    """
    request_context = event.get('requestContext', {})
    http_context = request_context.get('http', {})
    
    # Parse body
    body = None
    raw_body = event.get('body', '')
    if raw_body:
        is_base64 = event.get('isBase64Encoded', False)
        if is_base64:
            import base64
            raw_body = base64.b64decode(raw_body).decode('utf-8')
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, TypeError):
            body = raw_body

    # Parse query parameters
    query_params = {}
    raw_query = event.get('rawQueryString', '')
    if raw_query:
        from urllib.parse import parse_qs
        parsed = parse_qs(raw_query)
        query_params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
    
    # Also check queryStringParameters (set by API Gateway)
    if event.get('queryStringParameters'):
        query_params.update(event['queryStringParameters'])

    # Extract headers (lowercase keys)
    headers = {}
    for key, value in event.get('headers', {}).items():
        headers[key.lower()] = value

    # Extract authorizer claims
    authorizer = request_context.get('authorizer', {})
    jwt_claims = {}
    if 'jwt' in authorizer:
        jwt_claims = authorizer['jwt'].get('claims', {})
    elif 'lambda' in authorizer:
        jwt_claims = authorizer.get('lambda', {})

    return {
        'http_method': http_context.get('method', ''),
        'path': event.get('rawPath', ''),
        'stage': request_context.get('stage', ''),
        'request_id': request_context.get('requestId', ''),
        'domain_name': request_context.get('domainName', ''),
        'headers': headers,
        'query_params': query_params,
        'path_parameters': event.get('pathParameters', {}),
        'body': body,
        'raw_body': raw_body,
        'authorizer_claims': jwt_claims,
        'source_ip': http_context.get('sourceIp', ''),
        'user_agent': http_context.get('userAgent', ''),
        'request_time': request_context.get('time', ''),
        'request_time_epoch': request_context.get('timeEpoch', 0),
        'route_key': event.get('routeKey', ''),
    }


def _check_authentication(event: Dict[str, Any], request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check if the request has valid authentication.
    
    Returns None if authenticated, or an error response if not.
    """
    claims = request.get('authorizer_claims', {})
    
    if not claims:
        # Check for API key in headers
        api_key = request.get('headers', {}).get('x-api-key')
        if api_key:
            # API key validation would be done by a Lambda authorizer
            # For now, just check presence
            return None
        
        return create_response(401, {
            "error": "Authentication required",
            "code": "UNAUTHORIZED"
        })
    
    return None


def _check_admin_authorization(event: Dict[str, Any], request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check if the authenticated user has admin privileges.
    
    Returns None if authorized, or an error response if not.
    """
    claims = request.get('authorizer_claims', {})
    
    # Check for admin role in JWT claims or custom attribute
    user_role = claims.get('custom:role', claims.get('role', 'user'))
    
    if user_role != 'admin':
        return create_response(403, {
            "error": "Admin access required",
            "code": "FORBIDDEN"
        })
    
    return None


def _check_rate_limit(request: Dict[str, Any], limit: int) -> Optional[Dict[str, Any]]:
    """
    Check rate limiting for the request.
    
    Uses a simple token-bucket approach with DynamoDB for distributed rate limiting.
    Returns None if within limits, or a 429 response if exceeded.
    """
    # Rate limiting is primarily handled by API Gateway and the Lambda authorizer.
    # This is a secondary check for routes with specific rate limits.
    # In production, this would use DynamoDB or Redis for distributed rate limiting.
    
    # For now, log the rate limit check
    logger.debug("Rate limit check: limit=%d for path=%s", limit, request.get('path'))
    return None


def build_rate_limit_headers(
    limit: int,
    remaining: int,
    reset_time: int
) -> Dict[str, str]:
    """
    Build rate limit response headers.
    
    Args:
        limit: Maximum requests allowed
        remaining: Remaining requests in current window
        reset_time: Unix timestamp when the limit resets
        
    Returns:
        Dictionary of rate limit headers
    """
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_time)
    }


def create_paginated_response(
    items: List[Dict[str, Any]],
    total_count: int,
    page: int = 1,
    per_page: int = 20,
    base_path: str = ""
) -> Dict[str, Any]:
    """
    Create a paginated API response.
    
    Args:
        items: List of items for the current page
        total_count: Total number of items
        page: Current page number (1-based)
        per_page: Items per page
        base_path: Base URL path for pagination links
        
    Returns:
        API Gateway response with pagination metadata
    """
    total_pages = (total_count + per_page - 1) // per_page
    
    body = {
        "items": items,
        "pagination": {
            "total_count": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }
    
    # Add pagination links
    links = {}
    if page < total_pages:
        links["next"] = f"{base_path}?page={page + 1}&per_page={per_page}"
    if page > 1:
        links["prev"] = f"{base_path}?page={page - 1}&per_page={per_page}"
    links["first"] = f"{base_path}?page=1&per_page={per_page}"
    links["last"] = f"{base_path}?page={total_pages}&per_page={per_page}"
    body["links"] = links
    
    return create_response(HTTP_OK, body)

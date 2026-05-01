"""
AWS Client Service
------------------
HTTP client for proxying requests to the AWS API Gateway.
Handles authentication headers, request transformation, and error mapping.
"""

import json
import logging
import time
from typing import Optional, Dict, Any, List

import httpx
from botocore.auth import SigV4Auth
from botocore.credentials import Credentials
from botocore.awsrequest import AWSRequest

from ..core.config import settings

logger = logging.getLogger(__name__)


class APIGatewayClient:
    """
    Async HTTP client for communicating with AWS API Gateway.

    Supports two modes:
    1. Direct HTTP proxy — forwards requests to API Gateway URL
    2. AWS SigV4 signed requests — for IAM-authenticated endpoints
    """

    def __init__(self) -> None:
        self._base_url: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._aws_credentials: Optional[Credentials] = None

    async def initialize(self) -> None:
        """Initialize the HTTP client and AWS credentials."""
        self._base_url = settings.api_gateway_base_url

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.api_gateway_timeout),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AnimeQuoteGenerator-FastAPI/1.0",
            },
        )

        # Load AWS credentials for SigV4 signing (if configured)
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self._aws_credentials = Credentials(
                access_key=settings.aws_access_key_id,
                secret_key=settings.aws_secret_access_key,
            )
            logger.info("AWS credentials loaded for SigV4 request signing")

        logger.info(
            f"APIGatewayClient initialized — base_url={self._base_url or '(not configured)'}"
        )

    async def close(self) -> None:
        """Gracefully close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("APIGatewayClient closed")

    @property
    def is_configured(self) -> bool:
        """Check if the API Gateway URL is configured."""
        return bool(self._base_url)

    # ── Core HTTP Methods ────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """
        Send an HTTP request to API Gateway.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/v1/generate")
            json_body: Optional JSON request body
            params: Optional query parameters
            headers: Additional headers
            user_id: Optional user ID for request tracing
            api_key: Optional API key for public endpoints

        Returns:
            httpx.Response from API Gateway

        Raises:
            httpx.HTTPError: On network/connection errors
        """
        if not self._client or not self._base_url:
            raise RuntimeError("APIGatewayClient not initialized or not configured")

        url = f"{self._base_url}{path}"
        request_headers = dict(headers or {})

        # Add tracing headers
        if user_id:
            request_headers["X-User-ID"] = user_id
        if api_key:
            request_headers[settings.api_key_header_name] = api_key

        # Sign request with SigV4 if credentials available
        if self._aws_credentials:
            request_headers = self._sign_request(
                method=method,
                url=url,
                headers=request_headers,
                body=json.dumps(json_body) if json_body else "",
            )

        response = await self._client.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            headers=request_headers,
        )

        logger.debug(
            f"API Gateway {method} {path} → {response.status_code} "
            f"({response.elapsed.total_seconds():.3f}s)"
        )

        return response

    async def get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """Send GET request to API Gateway."""
        return await self._request(
            "GET", path, params=params, user_id=user_id, api_key=api_key
        )

    async def post(
        self,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """Send POST request to API Gateway."""
        return await self._request(
            "POST", path, json_body=json_body, user_id=user_id, api_key=api_key
        )

    async def put(
        self,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> httpx.Response:
        """Send PUT request to API Gateway."""
        return await self._request(
            "PUT", path, json_body=json_body, user_id=user_id
        )

    async def delete(
        self,
        path: str,
        *,
        user_id: Optional[str] = None,
    ) -> httpx.Response:
        """Send DELETE request to API Gateway."""
        return await self._request("DELETE", path, user_id=user_id)

    # ── SigV4 Signing ────────────────────────────────────────────

    def _sign_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: str,
    ) -> Dict[str, str]:
        """
        Sign an HTTP request with AWS SigV4.

        Args:
            method: HTTP method
            url: Full request URL
            headers: Existing headers
            body: Request body string

        Returns:
            Headers dict with SigV4 signature added
        """
        if not self._aws_credentials:
            return headers

        request = AWSRequest(method=method, url=url, data=body, headers=headers)
        sigv4 = SigV4Auth(
            self._aws_credentials,
            "execute-api",
            settings.aws_region,
        )
        sigv4.add_auth(request)

        signed_headers = dict(headers)
        signed_headers.update(dict(request.headers))
        return signed_headers

    # ── High-Level API Methods ───────────────────────────────────

    async def submit_generation(
        self, request_data: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """
        Submit a generation request to the pipeline.

        Args:
            request_data: Validated generation request body
            user_id: Authenticated user ID

        Returns:
            API Gateway response with job_id and status

        Raises:
            httpx.HTTPStatusError: On non-2xx responses
        """
        response = await self.post(
            "/v1/generate",
            json_body={**request_data, "user_id": user_id},
            user_id=user_id,
        )
        response.raise_for_status()
        return response.json()

    async def submit_batch_generation(
        self, batch_data: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """
        Submit a batch generation request.

        Args:
            batch_data: Validated batch request body
            user_id: Authenticated user ID

        Returns:
            API Gateway response with batch job details
        """
        response = await self.post(
            "/v1/generate/batch",
            json_body={**batch_data, "user_id": user_id},
            user_id=user_id,
        )
        response.raise_for_status()
        return response.json()

    async def get_job_status(
        self, job_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the status of a generation job.

        Args:
            job_id: Job identifier
            user_id: Optional user ID for authorization

        Returns:
            Job status details
        """
        response = await self.get(
            f"/v1/generate/{job_id}/status",
            user_id=user_id,
        )
        response.raise_for_status()
        return response.json()

    async def get_quotes(
        self,
        page: int = 1,
        page_size: int = 20,
        speech_type: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List generated quotes with pagination.

        Args:
            page: Page number
            page_size: Items per page
            speech_type: Optional filter by speech type
            api_key: API key for authentication

        Returns:
            Paginated quote list
        """
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if speech_type:
            params["speech_type"] = speech_type

        response = await self.get(
            "/v1/quotes",
            params=params,
            api_key=api_key,
        )
        response.raise_for_status()
        return response.json()

    async def get_quote(
        self, quote_id: str, api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get a specific quote by ID."""
        response = await self.get(
            f"/v1/quotes/{quote_id}",
            api_key=api_key,
        )
        response.raise_for_status()
        return response.json()

    async def get_random_quote(self) -> Dict[str, Any]:
        """Get a random quote (public endpoint)."""
        response = await self.get("/v1/quotes/random")
        response.raise_for_status()
        return response.json()

    async def get_health(self) -> Dict[str, Any]:
        """Check API Gateway health."""
        response = await self.get("/v1/health")
        response.raise_for_status()
        return response.json()

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get pipeline statistics."""
        response = await self.get("/v1/stats", user_id=user_id)
        response.raise_for_status()
        return response.json()

    async def get_config(self, user_id: str) -> Dict[str, Any]:
        """Get pipeline configuration (admin only)."""
        response = await self.get("/v1/config", user_id=user_id)
        response.raise_for_status()
        return response.json()


# ── Singleton Instance ───────────────────────────────────────────

api_gateway_client = APIGatewayClient()


async def get_api_gateway_client() -> APIGatewayClient:
    """FastAPI dependency: get the initialized API Gateway client."""
    if not api_gateway_client._client:
        await api_gateway_client.initialize()
    return api_gateway_client

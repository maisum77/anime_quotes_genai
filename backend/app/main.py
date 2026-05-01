"""
FastAPI Application
-------------------
Main application entry point for the Anime Quote Generator backend.
Configures middleware, routes, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .services.aws_client import api_gateway_client
from .services.websocket_manager import ws_manager
from .api.v1.endpoints import generation, quotes, auth, management, websocket

# ── Logging Configuration ────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=settings.log_format,
)
logger = logging.getLogger(__name__)


# ── Application Lifespan ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    Startup:
    - Initialize API Gateway HTTP client
    - Start WebSocket heartbeat monitoring

    Shutdown:
    - Close API Gateway HTTP client
    - Stop WebSocket manager
    """
    # ── Startup ──────────────────────────────────────────────
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"({settings.environment})"
    )

    await api_gateway_client.initialize()
    await ws_manager.start()

    logger.info("Application startup complete")

    yield  # Application runs here

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down application...")

    await ws_manager.stop()
    await api_gateway_client.close()

    logger.info("Application shutdown complete")


# ── Create FastAPI App ───────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "FastAPI backend for the Anime Quote Generator — "
        "proxies requests to the AWS Lambda processing pipeline "
        "via API Gateway, with JWT authentication and real-time "
        "WebSocket updates."
    ),
    docs_url=f"{settings.api_v1_prefix}/docs",
    redoc_url=f"{settings.api_v1_prefix}/redoc",
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Request Logging Middleware ────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their response status."""
    import time as _time

    start_time = _time.time()

    # Process the request
    response: Response = await call_next(request)

    # Calculate duration
    duration_ms = (_time.time() - start_time) * 1000

    # Log (skip health check spam in production)
    path = request.url.path
    if not (settings.is_production and path.endswith("/health")):
        logger.info(
            f"{request.method} {path} → {response.status_code} "
            f"({duration_ms:.1f}ms)"
        )

    # Add timing header
    response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"

    return response


# ── Exception Handlers ───────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions with a consistent error response."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        },
    )


# ── Register Routers ─────────────────────────────────────────────

# Authentication (no prefix — mounted at /api/v1/auth/*)
app.include_router(
    auth.router,
    prefix=settings.api_v1_prefix,
)

# Generation endpoints (mounted at /api/v1/generate/*)
app.include_router(
    generation.router,
    prefix=settings.api_v1_prefix,
)

# Quotes endpoints (mounted at /api/v1/quotes/*)
app.include_router(
    quotes.router,
    prefix=settings.api_v1_prefix,
)

# Management endpoints (mounted at /api/v1/health, /stats, /config)
app.include_router(
    management.router,
    prefix=settings.api_v1_prefix,
)

# WebSocket endpoint (mounted at /api/v1/ws)
app.include_router(
    websocket.router,
    prefix=settings.api_v1_prefix,
)


# ── Root Endpoint ────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to API documentation."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_v1_prefix}/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }

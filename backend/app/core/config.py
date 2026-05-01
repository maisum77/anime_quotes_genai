"""
Application Configuration
-------------------------
Centralized configuration management using pydantic-settings.
Supports environment variables, .env files, and sensible defaults.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    All settings can be overridden via environment variables with the
    prefix ANIME_QUOTE_ (e.g., ANIME_QUOTE_API_GATEWAY_URL).
    """

    model_config = SettingsConfigDict(
        env_prefix="ANIME_QUOTE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "Anime Quote Generator API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(
        default="dev",
        description="Deployment environment (dev/staging/prod)",
    )
    api_v1_prefix: str = "/api/v1"

    # ── Server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = Field(default=False, description="Enable auto-reload (dev only)")

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # ── JWT Authentication ───────────────────────────────────────
    jwt_secret_key: str = Field(
        default="change-me-in-production-use-a-strong-random-key",
        description="Secret key for JWT token signing",
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration in days"
    )
    jwt_issuer: str = "anime-quote-generator"
    jwt_audience: str = "anime-quote-generator-api"

    # ── API Key Authentication ───────────────────────────────────
    api_key_header_name: str = "X-API-Key"
    valid_api_keys: List[str] = Field(
        default=["dev-api-key-001"],
        description="Valid API keys for public endpoints",
    )

    # ── AWS API Gateway ──────────────────────────────────────────
    api_gateway_url: str = Field(
        default="",
        description="Base URL of the AWS API Gateway HTTP API",
    )
    api_gateway_stage: str = Field(
        default="dev",
        description="API Gateway deployment stage",
    )
    api_gateway_timeout: int = Field(
        default=30, description="HTTP timeout for API Gateway requests (seconds)"
    )

    # ── AWS Resources ────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key (prefer IAM roles)"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret key (prefer IAM roles)"
    )

    # S3
    s3_bucket_name: Optional[str] = Field(
        default=None, description="S3 bucket for generation outputs"
    )

    # SQS
    sqs_generation_queue_url: Optional[str] = Field(
        default=None, description="SQS generation queue URL"
    )
    sqs_postprocessing_queue_url: Optional[str] = Field(
        default=None, description="SQS postprocessing queue URL"
    )

    # DynamoDB
    dynamodb_jobs_table: str = Field(
        default="GenerationJobs", description="DynamoDB jobs table name"
    )
    dynamodb_history_table: str = Field(
        default="UserHistory", description="DynamoDB history table name"
    )
    dynamodb_metrics_table: str = Field(
        default="SystemMetrics", description="DynamoDB metrics table name"
    )

    # SNS
    sns_generation_complete_topic: Optional[str] = Field(
        default=None, description="SNS topic ARN for generation complete"
    )
    sns_generation_failed_topic: Optional[str] = Field(
        default=None, description="SNS topic ARN for generation failed"
    )

    # ── WebSocket ────────────────────────────────────────────────
    ws_heartbeat_interval: int = Field(
        default=30, description="WebSocket heartbeat interval (seconds)"
    )
    ws_max_connections: int = Field(
        default=100, description="Maximum concurrent WebSocket connections"
    )

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_per_minute: int = Field(
        default=60, description="API rate limit per minute per user"
    )
    rate_limit_burst: int = Field(
        default=10, description="Burst allowance for rate limiter"
    )

    # ── Redis (optional, for rate limiting & caching) ────────────
    redis_url: Optional[str] = Field(
        default=None, description="Redis URL for caching and rate limiting"
    )

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    @property
    def api_gateway_base_url(self) -> str:
        """Full API Gateway base URL including stage."""
        if not self.api_gateway_url:
            return ""
        stage = self.api_gateway_stage
        base = self.api_gateway_url.rstrip("/")
        return f"{base}/{stage}"

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"

    @property
    def is_development(self) -> bool:
        return self.environment == "dev"


# Singleton settings instance
settings = Settings()

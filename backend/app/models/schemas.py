"""
Pydantic Schemas
----------------
Request and response models for the FastAPI backend.
All models use Pydantic v2 with strict validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ────────────────────────────────────────────────────────

class SpeechType(str, Enum):
    """Supported anime speech types."""
    MOTIVATIONAL = "motivational"
    BATTLE = "battle"
    FRIENDSHIP = "friendship"
    DETERMINATION = "determination"
    VILLAIN = "villain"


class GenerationType(str, Enum):
    """Supported generation types."""
    SPEECH = "speech"
    DIALOGUE = "dialogue"


class JobStatus(str, Enum):
    """Job lifecycle statuses."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    IN_DLQ = "in_dlq"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SeverityLevel(str, Enum):
    """Notification severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ── Generation Request Models ────────────────────────────────────

class GenerationRequest(BaseModel):
    """Request model for single anime quote/speech generation."""
    generation_type: GenerationType = Field(
        default=GenerationType.SPEECH,
        description="Type of content to generate",
    )
    speech_type: SpeechType = Field(
        default=SpeechType.MOTIVATIONAL,
        description="Category of anime speech",
    )
    characters: List[str] = Field(
        default_factory=lambda: ["Hero", "Rival"],
        description="Character names for dialogue generation",
        max_length=10,
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt to guide generation",
        max_length=1000,
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1-2.0)",
    )
    max_length: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Maximum output length in tokens",
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for async completion notification",
    )

    @field_validator("characters")
    @classmethod
    def validate_characters(cls, v: List[str]) -> List[str]:
        """Ensure character names are non-empty strings."""
        return [c.strip() for c in v if c.strip()]


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""
    requests: List[GenerationRequest] = Field(
        ...,
        description="List of generation requests",
        min_length=1,
        max_length=10,
    )

    @field_validator("requests")
    @classmethod
    def validate_batch_size(cls, v: List[GenerationRequest]) -> List[GenerationRequest]:
        """Enforce maximum batch size."""
        if len(v) > 10:
            raise ValueError("Batch size cannot exceed 10 requests")
        return v


# ── Generation Response Models ───────────────────────────────────

class JobSubmittedResponse(BaseModel):
    """Response for a successfully submitted generation job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="queued", description="Initial job status")
    generation_type: str = Field(..., description="Type of generation requested")
    speech_type: str = Field(..., description="Speech type requested")
    estimated_time_seconds: int = Field(
        default=15, description="Estimated processing time"
    )
    status_url: str = Field(..., description="URL to check job status")
    created_at: str = Field(..., description="ISO 8601 timestamp")


class BatchJobSubmittedResponse(BaseModel):
    """Response for a batch generation submission."""
    batch_id: str = Field(..., description="Batch identifier")
    jobs: List[JobSubmittedResponse] = Field(
        ..., description="Individual job submissions"
    )
    total_jobs: int = Field(..., description="Total number of jobs in batch")


class JobStatusResponse(BaseModel):
    """Response for job status queries."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    generation_type: Optional[str] = None
    speech_type: Optional[str] = None
    generation_method: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    s3_key: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


# ── Content / Quote Models ───────────────────────────────────────

class QuoteResponse(BaseModel):
    """A single generated quote/speech."""
    id: str = Field(..., description="Quote identifier")
    content: str = Field(..., description="Generated speech text")
    speech_type: str = Field(..., description="Speech category")
    generation_type: str = Field(..., description="Generation type")
    generation_method: str = Field(..., description="Method used (gemini/gpt2/fallback)")
    characters: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class QuoteListResponse(BaseModel):
    """Paginated list of quotes."""
    quotes: List[QuoteResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of quotes")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=20, description="Items per page")
    has_next: bool = Field(default=False)


class RandomQuoteResponse(BaseModel):
    """Random quote response (public endpoint)."""
    content: str = Field(..., description="Random anime speech text")
    speech_type: str = Field(..., description="Speech category")
    generation_method: str = Field(
        default="fallback", description="How the quote was generated"
    )


# ── Authentication Models ────────────────────────────────────────

class LoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=8, max_length=200)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Ensure password meets minimum strength requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Valid refresh token")


class UserInfoResponse(BaseModel):
    """User profile information."""
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None


# ── Management / Stats Models ────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    timestamp: str = Field(..., description="Current ISO 8601 timestamp")
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services",
    )


class PipelineStatsResponse(BaseModel):
    """Pipeline statistics response."""
    total_jobs: int = Field(default=0)
    completed_jobs: int = Field(default=0)
    failed_jobs: int = Field(default=0)
    pending_jobs: int = Field(default=0)
    average_processing_time: Optional[float] = None
    generation_method_counts: Dict[str, int] = Field(default_factory=dict)
    speech_type_counts: Dict[str, int] = Field(default_factory=dict)
    recent_error_rate: Optional[float] = None


class ConfigurationResponse(BaseModel):
    """Pipeline configuration response (admin only)."""
    api_gateway_url: Optional[str] = None
    aws_region: str = Field(default="us-east-1")
    s3_bucket: Optional[str] = None
    dynamodb_jobs_table: str = Field(default="GenerationJobs")
    dynamodb_history_table: str = Field(default="UserHistory")
    sqs_generation_queue: Optional[str] = None
    sns_topics: Dict[str, Optional[str]] = Field(default_factory=dict)
    rate_limits: Dict[str, int] = Field(default_factory=dict)


# ── WebSocket Models ─────────────────────────────────────────────

class WSJobUpdate(BaseModel):
    """WebSocket message for job status updates."""
    type: str = Field(default="job_update")
    job_id: str
    status: JobStatus
    timestamp: str
    data: Optional[Dict[str, Any]] = None


class WSSubscription(BaseModel):
    """WebSocket subscription request."""
    action: str = Field(default="subscribe")
    job_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific job IDs to track (None = all)",
    )


class WSHeartbeat(BaseModel):
    """WebSocket heartbeat message."""
    type: str = Field(default="heartbeat")
    timestamp: str


# ── Error Models ─────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ValidationErrorDetail(BaseModel):
    """Individual validation error."""
    field: str
    message: str
    value: Optional[Any] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response with field details."""
    detail: str = "Validation error"
    errors: List[ValidationErrorDetail] = Field(default_factory=list)

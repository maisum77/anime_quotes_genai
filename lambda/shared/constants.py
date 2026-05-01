"""
Constants and configuration for Lambda functions
"""

# Speech types
SPEECH_TYPES = [
    "motivational",
    "battle", 
    "friendship",
    "determination",
    "villain"
]

# Generation types
GENERATION_TYPES = ["speech", "dialogue"]

# Default values
DEFAULT_SPEECH_TYPE = "motivational"
DEFAULT_GENERATION_TYPE = "speech"
DEFAULT_CHARACTERS = ["Hero", "Rival"]
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_LENGTH = 200
MAX_BATCH_SIZE = 10

# S3 bucket structure
S3_BUCKET_PREFIX = "anime-quote-generator"
S3_INPUTS_PREFIX = "inputs"
S3_OUTPUTS_PREFIX = "outputs"
S3_MODELS_PREFIX = "models"
S3_LOGS_PREFIX = "logs"
S3_CACHE_PREFIX = "cache"
S3_TEMPLATES_PREFIX = "templates"
S3_METRICS_PREFIX = "metrics"

# DynamoDB table names
DYNAMODB_JOBS_TABLE = "GenerationJobs"
DYNAMODB_HISTORY_TABLE = "UserHistory"
DYNAMODB_METRICS_TABLE = "SystemMetrics"

# SQS queue names
SQS_GENERATION_QUEUE = "generation-queue"
SQS_DLQ = "generation-dlq"
SQS_NOTIFICATION_QUEUE = "notification-queue"
SQS_POSTPROCESSING_QUEUE = "postprocessing-queue"

# SNS topic names
SNS_GENERATION_COMPLETE = "generation-complete"
SNS_GENERATION_FAILED = "generation-failed"
SNS_SYSTEM_ALERTS = "system-alerts"
SNS_USER_NOTIFICATIONS = "user-notifications"

# Job status constants
JOB_STATUS_PENDING = "pending"
JOB_STATUS_PREPROCESSING = "preprocessing"
JOB_STATUS_GENERATING = "generating"
JOB_STATUS_POSTPROCESSING = "postprocessing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_DLQ = "in_dlq"
JOB_STATUS_TIMEOUT = "timeout"
JOB_STATUS_CANCELLED = "cancelled"

# SNS event types
EVENT_GENERATION_COMPLETED = "generation_completed"
EVENT_GENERATION_FAILED = "generation_failed"
EVENT_PREPROCESSING_FAILED = "preprocessing_failed"
EVENT_POSTPROCESSING_FAILED = "postprocessing_failed"
EVENT_DLQ_OVERFLOW = "dlq_overflow"
EVENT_RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
EVENT_SYSTEM_ERROR = "system_error"
EVENT_MODEL_UNAVAILABLE = "model_unavailable"

# SNS severity levels
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"

# Environment variable names
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_S3_BUCKET = "S3_BUCKET"
ENV_JOBS_TABLE = "JOBS_TABLE"
ENV_GENERATION_QUEUE = "GENERATION_QUEUE"
ENV_DLQ = "DLQ_QUEUE"

# HTTP status codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504

# Error messages
ERROR_INVALID_JSON = "Invalid JSON body"
ERROR_INVALID_SPEECH_TYPE = f"Invalid speech_type. Must be one of: {SPEECH_TYPES}"
ERROR_INVALID_GENERATION_TYPE = f"Invalid generation_type. Must be one of: {GENERATION_TYPES}"
ERROR_MISSING_REQUIRED_FIELD = "Missing required field: {}"
ERROR_BATCH_SIZE_EXCEEDED = f"Batch size exceeds maximum of {MAX_BATCH_SIZE}"

# Fallback prompts (from original handler)
FALLBACK_PROMPTS = {
    "motivational": [
        "Listen everyone! True strength comes from within — never give up on your dreams!",
        "Stand up! Even if you fall a thousand times, rise a thousand and one!",
        "The path to victory is paved with the tears of those who never quit!",
    ],
    "battle": [
        "This battle isn't over! My power has no limits when I fight for my friends!",
        "You think you've won? I'm just getting started — prepare yourself!",
        "My true strength awakens when everything is on the line!",
    ],
    "friendship": [
        "We're not alone — together our bonds make us unstoppable!",
        "My friends are my greatest power; nothing can break what we've built!",
        "Side by side, we face every challenge that comes our way!",
    ],
    "determination": [
        "I made a promise I intend to keep, no matter the cost!",
        "Nothing will stop me from reaching my goal — not pain, not fear!",
        "Even if I fall, I'll keep moving forward until my last breath!",
    ],
    "villain": [
        "You fools cling to hope while the world crumbles beneath your feet!",
        "Power is the only truth — and I have claimed it all!",
        "Your resistance only delays the inevitable. Bow before me!",
    ],
}
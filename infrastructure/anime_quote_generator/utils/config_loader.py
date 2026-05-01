"""
Configuration loader for CDK environment-specific settings.

Reads context values from cdk.json and provides typed access
to environment configuration for all constructs.
"""

from typing import Any, Dict, Optional


class EnvironmentConfig:
    """Typed access to environment-specific CDK configuration."""

    def __init__(self, env_name: str, raw_config: Dict[str, Any]):
        self.env_name = env_name
        self._config = raw_config or {}

    # ── Region ──────────────────────────────────────────────────────
    @property
    def region(self) -> str:
        return self._config.get("region", "us-east-1")

    # ── Lambda settings ─────────────────────────────────────────────
    @property
    def lambda_memory(self) -> Dict[str, int]:
        return self._config.get("lambda_memory", {
            "preprocessing": 256,
            "generation": 1024,
            "postprocessing": 256,
        })

    @property
    def lambda_timeout(self) -> Dict[str, int]:
        return self._config.get("lambda_timeout", {
            "preprocessing": 30,
            "generation": 120,
            "postprocessing": 60,
        })

    def memory_for(self, function_name: str) -> int:
        """Return memory in MiB for a given Lambda function name."""
        return self.lambda_memory.get(function_name, 256)

    def timeout_for(self, function_name: str) -> int:
        """Return timeout in seconds for a given Lambda function name."""
        return self.lambda_timeout.get(function_name, 60)

    # ── Naming helpers ──────────────────────────────────────────────
    @property
    def resource_prefix(self) -> str:
        return f"anime-quote-gen-{self.env_name}"

    @property
    def s3_bucket_name(self) -> str:
        """Bucket name must be globally unique – caller should append account ID."""
        return f"anime-quote-generator-{self.env_name}"

    @property
    def dynamodb_jobs_table(self) -> str:
        return "GenerationJobs"

    @property
    def dynamodb_history_table(self) -> str:
        return "UserHistory"

    @property
    def dynamodb_metrics_table(self) -> str:
        return "SystemMetrics"

    # ── SQS / SNS names ────────────────────────────────────────────
    @property
    def sqs_generation_queue(self) -> str:
        return f"generation-queue-{self.env_name}"

    @property
    def sqs_postprocessing_queue(self) -> str:
        return f"postprocessing-queue-{self.env_name}"

    @property
    def sqs_notification_queue(self) -> str:
        return f"notification-queue-{self.env_name}"

    @property
    def sqs_dlq(self) -> str:
        return f"generation-dlq-{self.env_name}"

    @property
    def sns_generation_complete(self) -> str:
        return f"generation-complete-{self.env_name}"

    @property
    def sns_generation_failed(self) -> str:
        return f"generation-failed-{self.env_name}"

    @property
    def sns_system_alerts(self) -> str:
        return f"system-alerts-{self.env_name}"

    @property
    def sns_user_notifications(self) -> str:
        return f"user-notifications-{self.env_name}"

    # ── Feature flags ──────────────────────────────────────────────
    @property
    def enable_xray(self) -> bool:
        return self.env_name in ("staging", "prod")

    @property
    def retention_days(self) -> int:
        """CloudWatch log retention in days."""
        return 7 if self.env_name == "dev" else 30

    @property
    def is_production(self) -> bool:
        return self.env_name == "prod"

    # ── API Gateway ────────────────────────────────────────────────
    @property
    def api_throttle_rate_limit(self) -> int:
        return 20 if self.is_production else 100

    @property
    def api_throttle_burst_limit(self) -> int:
        return 40 if self.is_production else 200

    # ── Raw access ─────────────────────────────────────────────────
    @property
    def raw(self) -> Dict[str, Any]:
        return self._config

    def __repr__(self) -> str:
        return f"EnvironmentConfig(env={self.env_name!r}, region={self.region!r})"

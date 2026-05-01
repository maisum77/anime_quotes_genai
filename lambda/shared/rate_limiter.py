"""
Rate limiting utilities for API Gateway requests.

Implements token-bucket rate limiting using DynamoDB for
distributed rate tracking across Lambda instances.
"""
import os
import time
from typing import Dict, Any, Optional, Tuple
from .logging import logger


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter using DynamoDB for distributed state.
    
    Each API key or client gets a bucket with a configurable capacity
    and refill rate. Requests consume tokens; tokens refill over time.
    """

    def __init__(
        self,
        table_name: Optional[str] = None,
        default_capacity: int = 10,
        default_refill_rate: float = 2.0
    ):
        """
        Initialize the rate limiter.
        
        Args:
            table_name: DynamoDB table name for rate limit tracking
            default_capacity: Default bucket capacity (max burst)
            default_refill_rate: Default tokens refilled per second
        """
        self.table_name = table_name or os.environ.get('RATE_LIMIT_TABLE', 'anime-quote-rate-limits')
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self._dynamodb = None
        self._table = None

    @property
    def dynamodb(self):
        """Lazy DynamoDB resource initialization."""
        if self._dynamodb is None:
            import boto3
            self._dynamodb = boto3.resource('dynamodb')
        return self._dynamodb

    @property
    def table(self):
        """Lazy table reference initialization."""
        if self._table is None:
            self._table = self.dynamodb.Table(self.table_name)
        return self._table

    def check_rate_limit(
        self,
        identifier: str,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is within rate limits.
        
        Args:
            identifier: Unique identifier (API key, user ID, IP)
            capacity: Maximum tokens (burst capacity)
            refill_rate: Tokens refilled per second
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        capacity = capacity or self.default_capacity
        refill_rate = refill_rate or self.default_refill_rate
        current_time = time.time()

        try:
            # Get current bucket state
            response = self.table.get_item(
                Key={'identifier': identifier}
            )
            item = response.get('Item', {})

            last_refill = float(item.get('last_refill', current_time))
            current_tokens = float(item.get('current_tokens', capacity))

            # Calculate tokens to add based on time elapsed
            elapsed = current_time - last_refill
            tokens_to_add = elapsed * refill_rate
            current_tokens = min(capacity, current_tokens + tokens_to_add)

            # Check if request can be served
            if current_tokens >= 1:
                current_tokens -= 1
                is_allowed = True
            else:
                is_allowed = False

            # Calculate reset time
            if current_tokens < capacity:
                time_to_refill = (1 - current_tokens) / refill_rate
                reset_time = int(current_time + time_to_refill)
            else:
                reset_time = int(current_time)

            # Update DynamoDB
            self.table.put_item(
                Item={
                    'identifier': identifier,
                    'current_tokens': current_tokens,
                    'last_refill': current_time,
                    'capacity': capacity,
                    'refill_rate': refill_rate,
                    'ttl': int(current_time + 3600)  # Expire after 1 hour of inactivity
                }
            )

            rate_limit_info = {
                'limit': capacity,
                'remaining': int(current_tokens),
                'reset': reset_time,
                'retry_after': None if is_allowed else int((1 - current_tokens) / refill_rate)
            }

            return is_allowed, rate_limit_info

        except Exception as e:
            logger.error("Rate limit check failed: %s", e)
            # Fail open - allow request if rate limiter is unavailable
            return True, {
                'limit': capacity,
                'remaining': capacity,
                'reset': int(current_time + 60),
                'retry_after': None
            }

    def reset_bucket(self, identifier: str) -> None:
        """
        Reset rate limit bucket for an identifier.
        
        Args:
            identifier: Unique identifier to reset
        """
        try:
            self.table.delete_item(Key={'identifier': identifier})
        except Exception as e:
            logger.error("Failed to reset rate limit bucket: %s", e)

    def get_bucket_status(self, identifier: str) -> Dict[str, Any]:
        """
        Get current rate limit status for an identifier.
        
        Args:
            identifier: Unique identifier to check
            
        Returns:
            Dictionary with current bucket status
        """
        try:
            response = self.table.get_item(Key={'identifier': identifier})
            item = response.get('Item', {})
            
            if not item:
                return {
                    'identifier': identifier,
                    'current_tokens': self.default_capacity,
                    'capacity': self.default_capacity,
                    'status': 'empty'
                }
            
            current_time = time.time()
            last_refill = float(item.get('last_refill', current_time))
            current_tokens = float(item.get('current_tokens', 0))
            refill_rate = float(item.get('refill_rate', self.default_refill_rate))
            capacity = float(item.get('capacity', self.default_capacity))
            
            # Calculate current tokens
            elapsed = current_time - last_refill
            tokens_to_add = elapsed * refill_rate
            current_tokens = min(capacity, current_tokens + tokens_to_add)
            
            return {
                'identifier': identifier,
                'current_tokens': int(current_tokens),
                'capacity': int(capacity),
                'refill_rate': refill_rate,
                'status': 'healthy' if current_tokens > capacity * 0.2 else 'limited'
            }
        except Exception as e:
            logger.error("Failed to get bucket status: %s", e)
            return {'error': str(e)}


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using DynamoDB.
    
    Tracks request timestamps within a sliding time window
    for precise rate limiting without burst allowance.
    """

    def __init__(
        self,
        table_name: Optional[str] = None,
        window_seconds: int = 60,
        max_requests: int = 100
    ):
        """
        Initialize the sliding window rate limiter.
        
        Args:
            table_name: DynamoDB table name
            window_seconds: Time window in seconds
            max_requests: Maximum requests in the window
        """
        self.table_name = table_name or os.environ.get('RATE_LIMIT_TABLE', 'anime-quote-rate-limits')
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._dynamodb = None
        self._table = None

    @property
    def dynamodb(self):
        """Lazy DynamoDB resource initialization."""
        if self._dynamodb is None:
            import boto3
            self._dynamodb = boto3.resource('dynamodb')
        return self._dynamodb

    @property
    def table(self):
        """Lazy table reference initialization."""
        if self._table is None:
            self._table = self.dynamodb.Table(self.table_name)
        return self._table

    def check_rate_limit(
        self,
        identifier: str,
        window_seconds: Optional[int] = None,
        max_requests: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is within rate limits using sliding window.
        
        Args:
            identifier: Unique identifier
            window_seconds: Override window duration
            max_requests: Override max requests
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        window = window_seconds or self.window_seconds
        max_req = max_requests or self.max_requests
        current_time = time.time()
        window_start = current_time - window

        try:
            # Record this request
            request_id = f"{identifier}#{int(current_time * 1000)}"
            self.table.put_item(
                Item={
                    'identifier': identifier,
                    'request_id': request_id,
                    'timestamp': current_time,
                    'ttl': int(current_time + window + 60)
                }
            )

            # Count requests in the window
            response = self.table.query(
                KeyConditionExpression='identifier = :id',
                FilterExpression='#ts > :window_start',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':id': identifier,
                    ':window_start': window_start
                },
                Select='COUNT'
            )

            request_count = response.get('Count', 0)
            is_allowed = request_count <= max_req

            rate_limit_info = {
                'limit': max_req,
                'remaining': max(0, max_req - request_count),
                'reset': int(current_time + window),
                'retry_after': None if is_allowed else int(window),
                'window_seconds': window,
                'current_count': request_count
            }

            return is_allowed, rate_limit_info

        except Exception as e:
            logger.error("Sliding window rate limit check failed: %s", e)
            # Fail open
            return True, {
                'limit': max_req,
                'remaining': max_req,
                'reset': int(current_time + window),
                'retry_after': None
            }


# Rate limit tier configurations
RATE_LIMIT_TIERS = {
    'free': {
        'capacity': 5,
        'refill_rate': 0.5,  # 0.5 tokens/sec = 30/min
        'max_requests_per_minute': 30,
        'monthly_quota': 1000
    },
    'basic': {
        'capacity': 20,
        'refill_rate': 2.0,  # 2 tokens/sec = 120/min
        'max_requests_per_minute': 120,
        'monthly_quota': 10000
    },
    'premium': {
        'capacity': 100,
        'refill_rate': 10.0,  # 10 tokens/sec = 600/min
        'max_requests_per_minute': 600,
        'monthly_quota': 100000
    },
    'enterprise': {
        'capacity': 500,
        'refill_rate': 50.0,  # 50 tokens/sec = 3000/min
        'max_requests_per_minute': 3000,
        'monthly_quota': None  # Unlimited
    }
}


def get_rate_limiter_for_tier(tier: str) -> TokenBucketRateLimiter:
    """
    Get a rate limiter configured for a specific tier.
    
    Args:
        tier: Rate limit tier name
        
    Returns:
        Configured TokenBucketRateLimiter instance
    """
    config = RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS['free'])
    return TokenBucketRateLimiter(
        default_capacity=config['capacity'],
        default_refill_rate=config['refill_rate']
    )

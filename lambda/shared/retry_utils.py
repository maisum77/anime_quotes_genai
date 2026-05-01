"""
Retry utilities for Lambda functions
-----------------------------------
Provides retry mechanisms with exponential backoff for AWS service calls
and other operations that may fail transiently.
"""

import time
import random
import logging
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger()


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ):
        """
        Args:
            max_attempts: Maximum number of retry attempts (including initial)
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


def exponential_backoff(attempt: int, base_delay: float, max_delay: float, jitter: bool = True) -> float:
    """
    Calculate delay for exponential backoff.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        # Add up to 25% random jitter
        delay = delay * (0.75 + 0.25 * random.random())
    
    return delay


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration (uses defaults if None)
        
    Example:
        @retry_with_backoff()
        def call_external_api():
            # This will be retried up to 3 times with exponential backoff
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    # Don't sleep after the last attempt
                    if attempt < config.max_attempts - 1:
                        delay = exponential_backoff(
                            attempt, config.base_delay, config.max_delay, config.jitter
                        )
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: "
                            f"{str(e)}. Retrying in {delay:.2f}s"
                        )
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: "
                            f"{str(e)}"
                        )
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def retry_aws_service_call(
    func: Callable[..., T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0
) -> T:
    """
    Retry an AWS service call with exponential backoff.
    
    This is useful for transient AWS service errors like throttling,
    service unavailability, etc.
    
    Args:
        func: AWS service call function
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable AWS error
            error_code = getattr(e, 'response', {}).get('Error', {}).get('Code', '')
            
            # Common retryable AWS error codes
            retryable_codes = {
                'ThrottlingException',
                'Throttling',
                'ServiceUnavailable',
                'InternalError',
                'InternalFailure',
                'ProvisionedThroughputExceededException',
                'RequestLimitExceeded',
                'BandwidthLimitExceeded',
                'LimitExceededException',
                'TooManyRequestsException'
            }
            
            if error_code not in retryable_codes and attempt == 0:
                # Not a retryable error, fail immediately
                raise
            
            # Don't sleep after the last attempt
            if attempt < max_attempts - 1:
                delay = exponential_backoff(attempt, base_delay, max_delay, jitter=True)
                
                logger.warning(
                    f"AWS call attempt {attempt + 1}/{max_attempts} failed with {error_code}: "
                    f"{str(e)}. Retrying in {delay:.2f}s"
                )
                
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_attempts} AWS call attempts failed: {str(e)}"
                )
    
    # If we get here, all attempts failed
    raise last_exception


class CircuitBreaker:
    """
    Simple circuit breaker pattern implementation.
    
    Prevents repeated calls to a failing service by opening the circuit
    after a threshold of failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_attempts: int = 3
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting to close circuit
            half_open_max_attempts: Max attempts in half-open state before closing
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_attempts = half_open_max_attempts
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_attempts = 0
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If circuit is open or function fails
        """
        current_time = time.time()
        
        # Check circuit state
        if self.state == "OPEN":
            # Check if reset timeout has passed
            if current_time - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.half_open_attempts = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise RuntimeError(
                    f"Circuit breaker is OPEN. "
                    f"Next attempt in {self.reset_timeout - (current_time - self.last_failure_time):.1f}s"
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count and close circuit if needed
            if self.state == "HALF_OPEN":
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.half_open_max_attempts:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker transitioning to CLOSED state")
            
            elif self.state == "CLOSED":
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.state == "HALF_OPEN":
                # Failed in half-open state, go back to open
                self.state = "OPEN"
                logger.warning(f"Circuit breaker failed in HALF_OPEN state, transitioning to OPEN")
            
            elif self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self.state = "OPEN"
                logger.warning(f"Circuit breaker threshold reached, transitioning to OPEN")
            
            raise
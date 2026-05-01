"""
Logging and monitoring utilities for Lambda functions
"""
import json
import logging
import os
import time
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for Lambda functions
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def log_event(event: Dict[str, Any], context: Any = None) -> None:
    """
    Log Lambda event for debugging
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
    """
    try:
        event_str = json.dumps(event, default=str, indent=2)
        logger.info("Lambda Event: %s", event_str)
        
        if context:
            logger.info(
                "Lambda Context: function_name=%s, function_version=%s, "
                "invoked_function_arn=%s, memory_limit_in_mb=%s, aws_request_id=%s",
                context.function_name,
                context.function_version,
                context.invoked_function_arn,
                context.memory_limit_in_mb,
                context.aws_request_id
            )
    except Exception as e:
        logger.warning("Failed to log event: %s", e)

def log_metric(
    metric_name: str,
    value: float,
    unit: str = "Count",
    dimensions: Optional[Dict[str, str]] = None
) -> None:
    """
    Log custom metric to CloudWatch
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement (Count, Seconds, Bytes, etc.)
        dimensions: Metric dimensions
    """
    try:
        cloudwatch = boto3.client('cloudwatch')
        
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': time.time()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        # In production, you might want to batch metrics
        cloudwatch.put_metric_data(
            Namespace='AnimeQuoteGenerator',
            MetricData=[metric_data]
        )
        
        logger.debug("Logged metric %s=%s %s", metric_name, value, unit)
    except Exception as e:
        logger.warning("Failed to log metric %s: %s", metric_name, e)

def log_execution_time(start_time: float, operation: str) -> None:
    """
    Log execution time and emit CloudWatch metric
    
    Args:
        start_time: Start time from time.time()
        operation: Name of the operation being timed
    """
    execution_time = time.time() - start_time
    logger.info("%s execution time: %.3f seconds", operation, execution_time)
    
    # Log to CloudWatch
    log_metric(
        f"{operation}ExecutionTime",
        execution_time,
        unit="Seconds",
        dimensions={"Operation": operation}
    )

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log error with context
    
    Args:
        error: Exception object
        context: Additional context dictionary
    """
    error_msg = str(error)
    error_type = type(error).__name__
    
    logger.error(
        "Error: %s - %s",
        error_type,
        error_msg,
        exc_info=True
    )
    
    if context:
        logger.error("Error context: %s", json.dumps(context, default=str))
    
    # Log error metric
    log_metric(
        "Errors",
        1,
        unit="Count",
        dimensions={"ErrorType": error_type}
    )

def create_cloudwatch_alarm(
    alarm_name: str,
    metric_name: str,
    threshold: float,
    comparison_operator: str = "GreaterThanThreshold",
    evaluation_periods: int = 1,
    period: int = 300,
    statistic: str = "Average",
    dimensions: Optional[Dict[str, str]] = None
) -> None:
    """
    Create CloudWatch alarm (typically called from CDK, but available for runtime)
    
    Args:
        alarm_name: Name of the alarm
        metric_name: Metric to monitor
        threshold: Threshold value
        comparison_operator: Comparison operator
        evaluation_periods: Number of periods to evaluate
        period: Period in seconds
        statistic: Statistic to use
        dimensions: Metric dimensions
    """
    try:
        cloudwatch = boto3.client('cloudwatch')
        
        alarm_params = {
            'AlarmName': alarm_name,
            'ComparisonOperator': comparison_operator,
            'EvaluationPeriods': evaluation_periods,
            'MetricName': metric_name,
            'Namespace': 'AnimeQuoteGenerator',
            'Period': period,
            'Statistic': statistic,
            'Threshold': threshold,
            'ActionsEnabled': True,
            'AlarmDescription': f'Alarm for {metric_name}',
            'TreatMissingData': 'notBreaching'
        }
        
        if dimensions:
            alarm_params['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        # In production, alarms should be created via Infrastructure as Code
        # This is just for runtime emergency creation
        cloudwatch.put_metric_alarm(**alarm_params)
        logger.info("Created CloudWatch alarm: %s", alarm_name)
        
    except ClientError as e:
        logger.error("Failed to create CloudWatch alarm %s: %s", alarm_name, e)
    except Exception as e:
        logger.error("Unexpected error creating alarm: %s", e)
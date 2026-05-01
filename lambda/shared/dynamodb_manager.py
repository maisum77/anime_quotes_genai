"""
DynamoDB Metadata Manager
-------------------------
Comprehensive DynamoDB operations for the Anime Quote Generator pipeline:
1. Job tracking and lifecycle management
2. User history storage and retrieval
3. System metrics aggregation
4. Query and scan operations with pagination
5. Batch operations for high-throughput writes
6. TTL management for automatic data expiration

DynamoDB Table Schema:
    GenerationJobs (PK: job_id)
        - Tracks job lifecycle from preprocessing through postprocessing
        - Status transitions: pending → preprocessing → generating → postprocessing → completed/failed

    UserHistory (PK: user_id, SK: job_id)
        - Per-user generation history with GSI on timestamp for time-range queries
        - Enables user-facing dashboards and analytics

    SystemMetrics (PK: metric_name, SK: timestamp)
        - Time-series metrics for monitoring and alerting
        - GSI on metric_type for category-based queries
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from botocore.exceptions import ClientError

from .aws_clients import clients
from .constants import (
    DYNAMODB_JOBS_TABLE, DYNAMODB_HISTORY_TABLE, DYNAMODB_METRICS_TABLE,
    JOB_STATUS_PENDING, JOB_STATUS_PREPROCESSING, JOB_STATUS_GENERATING,
    JOB_STATUS_POSTPROCESSING, JOB_STATUS_COMPLETED, JOB_STATUS_FAILED,
    JOB_STATUS_DLQ, JOB_STATUS_TIMEOUT, JOB_STATUS_CANCELLED
)

logger = logging.getLogger(__name__)

# Valid status transitions
STATUS_TRANSITIONS = {
    JOB_STATUS_PENDING: [JOB_STATUS_PREPROCESSING, JOB_STATUS_CANCELLED],
    JOB_STATUS_PREPROCESSING: [JOB_STATUS_GENERATING, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED],
    JOB_STATUS_GENERATING: [JOB_STATUS_POSTPROCESSING, JOB_STATUS_FAILED, JOB_STATUS_TIMEOUT],
    JOB_STATUS_POSTPROCESSING: [JOB_STATUS_COMPLETED, JOB_STATUS_FAILED],
    JOB_STATUS_FAILED: [JOB_STATUS_PENDING],  # Allow retry from failed
}

# Default TTL values (in seconds)
DEFAULT_JOB_TTL = 30 * 24 * 3600       # 30 days
DEFAULT_HISTORY_TTL = 90 * 24 * 3600   # 90 days
DEFAULT_METRICS_TTL = 365 * 24 * 3600  # 1 year


def _get_table_name(table_name: str) -> str:
    """
    Resolve table name with environment prefix.

    Args:
        table_name: Base table name

    Returns:
        Environment-prefixed table name
    """
    env = os.environ.get('ENVIRONMENT', 'dev')
    return f"{table_name}-{env}" if env != 'prod' else table_name


def _get_table_resource(table_name: str):
    """
    Get a DynamoDB Table resource.

    Args:
        table_name: Table name

    Returns:
        DynamoDB Table resource
    """
    full_name = _get_table_name(table_name)
    return clients.dynamodb.Table(full_name)


# ---------------------------------------------------------------------------
# Job Tracking Operations
# ---------------------------------------------------------------------------

def create_job(
    job_id: str,
    request_data: Dict[str, Any],
    user_id: Optional[str] = None,
    request_type: str = "single"
) -> Dict[str, Any]:
    """
    Create a new job record in DynamoDB.

    Args:
        job_id: Unique job identifier
        request_data: Original request payload
        user_id: Optional user identifier
        request_type: Type of request ('single' or 'batch')

    Returns:
        Created job item
    """
    now = int(time.time())

    item = {
        'job_id': job_id,
        'status': JOB_STATUS_PENDING,
        'request_type': request_type,
        'request_data': json.dumps(request_data, default=str),
        'created_at': now,
        'updated_at': now,
        'ttl': now + DEFAULT_JOB_TTL,
    }

    if user_id:
        item['user_id'] = user_id

    # Add request metadata
    item['speech_type'] = request_data.get('speech_type', 'motivational')
    item['generation_type'] = request_data.get('generation_type', 'speech')

    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)
        table.put_item(
            Item=item,
            ConditionExpression='attribute_not_exists(job_id)'
        )

        logger.info("Created job %s with status %s", job_id, JOB_STATUS_PENDING)
        return item

    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            logger.warning("Job %s already exists", job_id)
            raise ValueError(f"Job {job_id} already exists")
        logger.error("Failed to create job %s: %s", job_id, e)
        raise Exception(f"Failed to create job: {e}")


def update_job_status(
    job_id: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update job status with optional metadata.

    Performs a conditional update to enforce valid status transitions.

    Args:
        job_id: Job identifier
        status: New status value
        metadata: Optional additional metadata to merge

    Returns:
        Updated job attributes

    Raises:
        Exception: If update fails or transition is invalid
    """
    now = int(time.time())

    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)

        update_expr = "SET #status = :status, updated_at = :updated_at"
        expr_attr_names = {
            '#status': 'status'
        }
        expr_attr_values: Dict[str, Any] = {
            ':status': status,
            ':updated_at': now
        }

        # Add metadata fields
        if metadata:
            for key, value in metadata.items():
                attr_name = f"meta_{key}"
                update_expr += f", {attr_name} = :{attr_name}"
                expr_attr_values[f":{attr_name}"] = value

        # Add status-specific timestamps
        if status == JOB_STATUS_PREPROCESSING:
            update_expr += ", preprocessing_started_at = :ts"
            expr_attr_values[':ts'] = now
        elif status == JOB_STATUS_GENERATING:
            update_expr += ", generation_started_at = :ts"
            expr_attr_values[':ts'] = now
        elif status == JOB_STATUS_POSTPROCESSING:
            update_expr += ", postprocessing_started_at = :ts"
            expr_attr_values[':ts'] = now
        elif status == JOB_STATUS_COMPLETED:
            update_expr += ", completed_at = :ts"
            expr_attr_values[':ts'] = now
        elif status == JOB_STATUS_FAILED:
            update_expr += ", failed_at = :ts"
            expr_attr_values[':ts'] = now

        response = table.update_item(
            Key={'job_id': job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values,
            ReturnValues='ALL_NEW'
        )

        logger.info("Updated job %s status to %s", job_id, status)
        return response.get('Attributes', {})

    except ClientError as e:
        logger.error("Failed to update job %s: %s", job_id, e)
        raise Exception(f"Failed to update job status: {e}")


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a job record by job ID.

    Args:
        job_id: Job identifier

    Returns:
        Job record dict or None if not found
    """
    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)
        response = table.get_item(Key={'job_id': job_id})

        item = response.get('Item')
        if item and 'request_data' in item and isinstance(item['request_data'], str):
            try:
                item['request_data'] = json.loads(item['request_data'])
            except json.JSONDecodeError:
                pass

        return item

    except ClientError as e:
        logger.error("Failed to get job %s: %s", job_id, e)
        raise Exception(f"Failed to get job: {e}")


def get_job_status(job_id: str) -> Optional[str]:
    """
    Get just the status of a job.

    Args:
        job_id: Job identifier

    Returns:
        Status string or None if job not found
    """
    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)
        response = table.get_item(
            Key={'job_id': job_id},
            ProjectionExpression='#status',
            ExpressionAttributeNames={'#status': 'status'}
        )

        item = response.get('Item')
        return item.get('status') if item else None

    except ClientError as e:
        logger.error("Failed to get job status for %s: %s", job_id, e)
        return None


def delete_job(job_id: str) -> bool:
    """
    Delete a job record.

    Args:
        job_id: Job identifier

    Returns:
        True if deletion succeeded
    """
    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)
        table.delete_item(Key={'job_id': job_id})
        logger.info("Deleted job %s", job_id)
        return True
    except ClientError as e:
        logger.error("Failed to delete job %s: %s", job_id, e)
        return False


def list_jobs_by_status(
    status: str,
    limit: int = 50,
    last_key: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    List jobs filtered by status using a GSI query.

    Args:
        status: Job status to filter by
        limit: Maximum number of items to return
        last_key: Pagination token from previous query

    Returns:
        Dict with 'items' list and 'last_evaluated_key' for pagination
    """
    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)

        kwargs: Dict[str, Any] = {
            'IndexName': 'status-index',
            'KeyConditionExpression': '#status = :status',
            'ExpressionAttributeNames': {'#status': 'status'},
            'ExpressionAttributeValues': {':status': status},
            'Limit': limit,
            'ScanIndexForward': False  # Most recent first
        }

        if last_key:
            kwargs['ExclusiveStartKey'] = last_key

        response = table.query(**kwargs)

        items = response.get('Items', [])

        return {
            'items': items,
            'count': response.get('Count', 0),
            'last_evaluated_key': response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        logger.error("Failed to list jobs by status %s: %s", status, e)
        raise Exception(f"Failed to list jobs: {e}")


def list_jobs_by_user(
    user_id: str,
    limit: int = 50,
    last_key: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    List jobs for a specific user.

    Args:
        user_id: User identifier
        limit: Maximum items to return
        last_key: Pagination token

    Returns:
        Dict with 'items' list and pagination info
    """
    try:
        table = _get_table_resource(DYNAMODB_JOBS_TABLE)

        kwargs: Dict[str, Any] = {
            'IndexName': 'user_id-index',
            'KeyConditionExpression': 'user_id = :user_id',
            'ExpressionAttributeValues': {':user_id': user_id},
            'Limit': limit,
            'ScanIndexForward': False
        }

        if last_key:
            kwargs['ExclusiveStartKey'] = last_key

        response = table.query(**kwargs)

        return {
            'items': response.get('Items', []),
            'count': response.get('Count', 0),
            'last_evaluated_key': response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        logger.error("Failed to list jobs for user %s: %s", user_id, e)
        raise Exception(f"Failed to list user jobs: {e}")


def get_jobs_count_by_status() -> Dict[str, int]:
    """
    Get count of jobs in each status category.

    Returns:
        Dict mapping status to count
    """
    counts = {}
    for status in [
        JOB_STATUS_PENDING, JOB_STATUS_PREPROCESSING,
        JOB_STATUS_GENERATING, JOB_STATUS_POSTPROCESSING,
        JOB_STATUS_COMPLETED, JOB_STATUS_FAILED
    ]:
        try:
            result = list_jobs_by_status(status, limit=1)
            counts[status] = result.get('count', 0)
        except Exception:
            counts[status] = 0

    return counts


# ---------------------------------------------------------------------------
# User History Operations
# ---------------------------------------------------------------------------

def store_user_history(
    history_item: Dict[str, Any]
) -> bool:
    """
    Store a generation record in user history.

    Args:
        history_item: History item with user_id, job_id, and result data

    Returns:
        True if storage succeeded
    """
    try:
        table = _get_table_resource(DYNAMODB_HISTORY_TABLE)

        # Ensure required fields
        now = int(time.time())
        item = {
            'user_id': history_item.get('user_id', 'anonymous'),
            'job_id': history_item.get('job_id'),
            'timestamp': history_item.get('timestamp', now),
            'created_at': now,
            'ttl': now + DEFAULT_HISTORY_TTL,
        }

        # Add optional fields
        if 'request' in history_item:
            item['request'] = json.dumps(history_item['request'], default=str)
        if 'result' in history_item:
            item['result'] = json.dumps(history_item['result'], default=str)

        # Add any extra fields
        for key in ['speech_type', 'generation_type', 'generation_method', 'word_count']:
            if key in history_item:
                item[key] = history_item[key]

        table.put_item(Item=item)

        logger.info(
            "Stored history for user %s, job %s",
            item['user_id'], item['job_id']
        )
        return True

    except ClientError as e:
        logger.error("Failed to store user history: %s", e)
        raise Exception(f"Failed to store user history: {e}")


def get_user_history(
    user_id: str,
    limit: int = 50,
    last_key: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get generation history for a user.

    Args:
        user_id: User identifier
        limit: Maximum items to return
        last_key: Pagination token

    Returns:
        Dict with 'items' list and pagination info
    """
    try:
        table = _get_table_resource(DYNAMODB_HISTORY_TABLE)

        kwargs: Dict[str, Any] = {
            'KeyConditionExpression': 'user_id = :user_id',
            'ExpressionAttributeValues': {':user_id': user_id},
            'Limit': limit,
            'ScanIndexForward': False  # Most recent first
        }

        if last_key:
            kwargs['ExclusiveStartKey'] = last_key

        response = table.query(**kwargs)

        items = response.get('Items', [])

        # Deserialize JSON fields
        for item in items:
            if 'request' in item and isinstance(item['request'], str):
                try:
                    item['request'] = json.loads(item['request'])
                except json.JSONDecodeError:
                    pass
            if 'result' in item and isinstance(item['result'], str):
                try:
                    item['result'] = json.loads(item['result'])
                except json.JSONDecodeError:
                    pass

        return {
            'items': items,
            'count': response.get('Count', 0),
            'last_evaluated_key': response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        logger.error("Failed to get history for user %s: %s", user_id, e)
        raise Exception(f"Failed to get user history: {e}")


def get_user_stats(user_id: str) -> Dict[str, Any]:
    """
    Get aggregated statistics for a user.

    Args:
        user_id: User identifier

    Returns:
        Dict with user statistics
    """
    try:
        table = _get_table_resource(DYNAMODB_HISTORY_TABLE)

        response = table.query(
            KeyConditionExpression='user_id = :user_id',
            ExpressionAttributeValues={':user_id': user_id},
            Select='COUNT'
        )

        total_generations = response.get('Count', 0)

        # Get recent generations for detailed stats
        recent = get_user_history(user_id, limit=100)
        items = recent.get('items', [])

        speech_types: Dict[str, int] = {}
        generation_methods: Dict[str, int] = {}
        total_words = 0

        for item in items:
            st = item.get('speech_type', 'unknown')
            speech_types[st] = speech_types.get(st, 0) + 1

            gm = item.get('generation_method', 'unknown')
            generation_methods[gm] = generation_methods.get(gm, 0) + 1

            total_words += item.get('word_count', 0)

        return {
            'user_id': user_id,
            'total_generations': total_generations,
            'speech_type_distribution': speech_types,
            'generation_method_distribution': generation_methods,
            'total_words_generated': total_words,
            'average_words_per_generation': round(total_words / max(len(items), 1), 1)
        }

    except ClientError as e:
        logger.error("Failed to get stats for user %s: %s", user_id, e)
        raise Exception(f"Failed to get user stats: {e}")


def delete_user_history(
    user_id: str,
    job_id: Optional[str] = None
) -> bool:
    """
    Delete user history entries.

    If job_id is specified, deletes only that entry.
    Otherwise, deletes all history for the user.

    Args:
        user_id: User identifier
        job_id: Optional specific job to delete

    Returns:
        True if deletion succeeded
    """
    try:
        table = _get_table_resource(DYNAMODB_HISTORY_TABLE)

        if job_id:
            table.delete_item(Key={'user_id': user_id, 'job_id': job_id})
            logger.info("Deleted history entry for user %s, job %s", user_id, job_id)
        else:
            # Delete all history for user (batch)
            response = table.query(
                KeyConditionExpression='user_id = :user_id',
                ExpressionAttributeValues={':user_id': user_id},
                ProjectionExpression='user_id, job_id'
            )

            with table.batch_writer() as batch:
                for item in response.get('Items', []):
                    batch.delete_item(Key={'user_id': item['user_id'], 'job_id': item['job_id']})

            logger.info("Deleted all history for user %s", user_id)

        return True

    except ClientError as e:
        logger.error("Failed to delete history for user %s: %s", user_id, e)
        return False


# ---------------------------------------------------------------------------
# System Metrics Operations
# ---------------------------------------------------------------------------

def record_metric(
    metric_name: str,
    value: float,
    unit: str = "Count",
    dimensions: Optional[Dict[str, str]] = None,
    metric_type: str = "custom"
) -> bool:
    """
    Record a time-series metric in DynamoDB.

    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement (Count, Seconds, Percent, etc.)
        dimensions: Optional dimension key-value pairs
        metric_type: Metric category (custom, system, business)

    Returns:
        True if recording succeeded
    """
    now = int(time.time())

    try:
        table = _get_table_resource(DYNAMODB_METRICS_TABLE)

        item = {
            'metric_name': metric_name,
            'timestamp': now,
            'value': value,
            'unit': unit,
            'metric_type': metric_type,
            'ttl': now + DEFAULT_METRICS_TTL
        }

        if dimensions:
            item['dimensions'] = json.dumps(dimensions)

        table.put_item(Item=item)

        logger.debug("Recorded metric %s = %s %s", metric_name, value, unit)
        return True

    except ClientError as e:
        logger.error("Failed to record metric %s: %s", metric_name, e)
        return False


def get_metrics(
    metric_name: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get time-series metrics for a given metric name.

    Args:
        metric_name: Name of the metric
        start_time: Start timestamp (defaults to 1 hour ago)
        end_time: End timestamp (defaults to now)
        limit: Maximum number of data points

    Returns:
        List of metric data points
    """
    now = int(time.time())
    if start_time is None:
        start_time = now - 3600  # 1 hour ago
    if end_time is None:
        end_time = now

    try:
        table = _get_table_resource(DYNAMODB_METRICS_TABLE)

        response = table.query(
            KeyConditionExpression='metric_name = :name AND #ts BETWEEN :start AND :end',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={
                ':name': metric_name,
                ':start': start_time,
                ':end': end_time
            },
            Limit=limit,
            ScanIndexForward=False  # Most recent first
        )

        items = response.get('Items', [])

        # Deserialize dimensions
        for item in items:
            if 'dimensions' in item and isinstance(item['dimensions'], str):
                try:
                    item['dimensions'] = json.loads(item['dimensions'])
                except json.JSONDecodeError:
                    pass

        return items

    except ClientError as e:
        logger.error("Failed to get metrics for %s: %s", metric_name, e)
        raise Exception(f"Failed to get metrics: {e}")


def get_metric_summary(
    metric_name: str,
    period_seconds: int = 3600
) -> Dict[str, Any]:
    """
    Get aggregated summary for a metric over a time period.

    Args:
        metric_name: Name of the metric
        period_seconds: Time period in seconds (default 1 hour)

    Returns:
        Dict with sum, avg, min, max, count, and sample_count
    """
    now = int(time.time())
    start_time = now - period_seconds

    try:
        data_points = get_metrics(metric_name, start_time, now, limit=1000)

        if not data_points:
            return {
                'metric_name': metric_name,
                'period_seconds': period_seconds,
                'count': 0,
                'sum': 0,
                'avg': 0,
                'min': 0,
                'max': 0
            }

        values = [dp.get('value', 0) for dp in data_points]

        return {
            'metric_name': metric_name,
            'period_seconds': period_seconds,
            'count': len(values),
            'sum': round(sum(values), 4),
            'avg': round(sum(values) / len(values), 4),
            'min': round(min(values), 4),
            'max': round(max(values), 4),
            'data_points': len(data_points)
        }

    except Exception as e:
        logger.error("Failed to get metric summary for %s: %s", metric_name, e)
        return {
            'metric_name': metric_name,
            'period_seconds': period_seconds,
            'error': str(e)
        }


def list_metric_names() -> List[str]:
    """
    List all unique metric names in the metrics table.

    Returns:
        List of metric name strings
    """
    try:
        table = _get_table_resource(DYNAMODB_METRICS_TABLE)

        response = table.scan(
            ProjectionExpression='metric_name',
            Select='SPECIFIC_ATTRIBUTES'
        )

        names = list(set(item['metric_name'] for item in response.get('Items', [])))
        return sorted(names)

    except ClientError as e:
        logger.error("Failed to list metric names: %s", e)
        return []


# ---------------------------------------------------------------------------
# Batch Operations
# ---------------------------------------------------------------------------

def batch_write_jobs(
    items: List[Dict[str, Any]],
    table_name: str = DYNAMODB_JOBS_TABLE
) -> Dict[str, Any]:
    """
    Batch write job items to DynamoDB.

    Args:
        items: List of job items to write
        table_name: Target table name

    Returns:
        Dict with 'written' count and 'errors' list
    """
    written = 0
    errors = []

    try:
        table = _get_table_resource(table_name)

        with table.batch_writer() as batch:
            for item in items:
                try:
                    # Ensure JSON serialization for complex fields
                    if 'request_data' in item and isinstance(item['request_data'], dict):
                        item['request_data'] = json.dumps(item['request_data'], default=str)

                    batch.put_item(Item=item)
                    written += 1
                except Exception as e:
                    errors.append({
                        'job_id': item.get('job_id', 'unknown'),
                        'error': str(e)
                    })

        logger.info("Batch wrote %d items to %s", written, table_name)
        return {'written': written, 'errors': errors}

    except ClientError as e:
        logger.error("Batch write failed for %s: %s", table_name, e)
        raise Exception(f"Batch write failed: {e}")


def batch_write_history(
    items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Batch write user history items.

    Args:
        items: List of history items

    Returns:
        Dict with 'written' count and 'errors' list
    """
    written = 0
    errors = []

    try:
        table = _get_table_resource(DYNAMODB_HISTORY_TABLE)

        with table.batch_writer() as batch:
            for item in items:
                try:
                    now = int(time.time())
                    if 'ttl' not in item:
                        item['ttl'] = now + DEFAULT_HISTORY_TTL
                    if 'created_at' not in item:
                        item['created_at'] = now

                    # Serialize complex fields
                    for field in ['request', 'result']:
                        if field in item and isinstance(item[field], dict):
                            item[field] = json.dumps(item[field], default=str)

                    batch.put_item(Item=item)
                    written += 1
                except Exception as e:
                    errors.append({
                        'user_id': item.get('user_id', 'unknown'),
                        'job_id': item.get('job_id', 'unknown'),
                        'error': str(e)
                    })

        logger.info("Batch wrote %d history items", written)
        return {'written': written, 'errors': errors}

    except ClientError as e:
        logger.error("Batch history write failed: %s", e)
        raise Exception(f"Batch history write failed: {e}")


# ---------------------------------------------------------------------------
# Utility Operations
# ---------------------------------------------------------------------------

def scan_table(
    table_name: str,
    filter_expression: Optional[str] = None,
    expression_values: Optional[Dict[str, Any]] = None,
    expression_names: Optional[Dict[str, str]] = None,
    limit: int = 100,
    projection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a scan operation on a DynamoDB table.

    Use sparingly - scans are expensive on large tables.

    Args:
        table_name: Table name
        filter_expression: Optional filter expression
        expression_values: Expression attribute values
        expression_names: Expression attribute names
        limit: Maximum items to return
        projection: Optional projection expression

    Returns:
        Dict with 'items' list and 'count'
    """
    try:
        table = _get_table_resource(table_name)

        kwargs: Dict[str, Any] = {'Limit': limit}

        if filter_expression:
            kwargs['FilterExpression'] = filter_expression
        if expression_values:
            kwargs['ExpressionAttributeValues'] = expression_values
        if expression_names:
            kwargs['ExpressionAttributeNames'] = expression_names
        if projection:
            kwargs['ProjectionExpression'] = projection

        response = table.scan(**kwargs)

        return {
            'items': response.get('Items', []),
            'count': response.get('Count', 0),
            'scanned_count': response.get('ScannedCount', 0)
        }

    except ClientError as e:
        logger.error("Scan failed for %s: %s", table_name, e)
        raise Exception(f"Table scan failed: {e}")


def get_table_item_count(table_name: str) -> Optional[int]:
    """
    Get approximate item count for a DynamoDB table.

    Uses DescribeTable which returns an approximate count.

    Args:
        table_name: Table name

    Returns:
        Approximate item count or None
    """
    try:
        full_name = _get_table_name(table_name)
        response = clients.dynamodb.meta.client.describe_table(TableName=full_name)
        return response.get('Table', {}).get('ItemCount', 0)
    except ClientError as e:
        logger.error("Failed to get item count for %s: %s", table_name, e)
        return None


def get_dynamodb_client():
    """Get the underlying DynamoDB client for advanced operations."""
    return clients.dynamodb.meta.client


def get_dynamodb_resource():
    """Get the underlying DynamoDB resource for advanced operations."""
    return clients.dynamodb

"""
SQS Queue Manager
-----------------
Comprehensive SQS operations for the Anime Quote Generator pipeline:
1. Message sending with attributes and deduplication
2. Message receiving with batch processing
3. Dead-letter queue management
4. Queue monitoring and metrics
5. Message lifecycle management (delete, visibility timeout)
6. Batch operations for high-throughput processing

Queue Architecture:
    generation-queue       → Main queue for generation requests (preprocessing → generation)
    generation-dlq         → Dead-letter queue for failed generation attempts
    notification-queue     → Queue for async notification delivery
    postprocessing-queue   → Queue for postprocessing tasks (generation → postprocessing)
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from botocore.exceptions import ClientError

from .aws_clients import clients
from .constants import (
    SQS_GENERATION_QUEUE, SQS_DLQ, SQS_NOTIFICATION_QUEUE,
    SQS_POSTPROCESSING_QUEUE
)

logger = logging.getLogger(__name__)

# Message attribute types
MSG_ATTR_STRING = "String"
MSG_ATTR_NUMBER = "Number"
MSG_ATTR_BINARY = "Binary"

# Default configuration
DEFAULT_VISIBILITY_TIMEOUT = 300  # 5 minutes
DEFAULT_MAX_RECEIVE_COUNT = 3     # Max retries before DLQ
DEFAULT_MESSAGE_RETENTION = 1209600  # 14 days in seconds
DEFAULT_BATCH_SIZE = 10           # Max SQS batch size
MAX_SQS_BATCH_SIZE = 10          # AWS SQS limit


def _get_queue_url(queue_name: str) -> str:
    """
    Get SQS queue URL from queue name.

    Args:
        queue_name: SQS queue name

    Returns:
        Queue URL string

    Raises:
        Exception: If queue URL cannot be resolved
    """
    try:
        response = clients.sqs.get_queue_url(QueueName=queue_name)
        return response['QueueUrl']
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'AWS.SimpleQueueService.NonExistentQueue':
            logger.error("Queue does not exist: %s", queue_name)
            raise Exception(f"SQS queue does not exist: {queue_name}")
        logger.error("Failed to get queue URL for %s: %s", queue_name, e)
        raise Exception(f"Failed to get queue URL: {e}")


def _get_queue_url_from_env(env_var: str, default_name: str) -> str:
    """
    Get queue URL from environment variable or fall back to queue name lookup.

    Args:
        env_var: Environment variable name containing queue URL
        default_name: Default queue name if env var not set

    Returns:
        Queue URL string
    """
    queue_url = os.environ.get(env_var)
    if queue_url:
        return queue_url
    return _get_queue_url(default_name)


def _build_message_attributes(
    message_type: Optional[str] = None,
    job_id: Optional[str] = None,
    priority: Optional[str] = None,
    custom_attrs: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Dict[str, str]]:
    """
    Build SQS message attributes dict.

    Args:
        message_type: Type of message (e.g., 'GenerationRequest', 'PostprocessingRequest')
        job_id: Job identifier for tracing
        priority: Priority level ('high', 'normal', 'low')
        custom_attrs: Additional custom attributes

    Returns:
        SQS message attributes dictionary
    """
    attrs: Dict[str, Dict[str, str]] = {}

    if message_type:
        attrs['MessageType'] = {
            'DataType': MSG_ATTR_STRING,
            'StringValue': message_type
        }

    if job_id:
        attrs['JobId'] = {
            'DataType': MSG_ATTR_STRING,
            'StringValue': job_id
        }

    if priority:
        attrs['Priority'] = {
            'DataType': MSG_ATTR_STRING,
            'StringValue': priority
        }

    attrs['Timestamp'] = {
        'DataType': MSG_ATTR_STRING,
        'StringValue': datetime.utcnow().isoformat() + "Z"
    }

    if custom_attrs:
        attrs.update(custom_attrs)

    return attrs


# ---------------------------------------------------------------------------
# Send Operations
# ---------------------------------------------------------------------------

def send_message(
    message: Dict[str, Any],
    queue_name: str = SQS_GENERATION_QUEUE,
    message_type: Optional[str] = None,
    job_id: Optional[str] = None,
    priority: Optional[str] = None,
    delay_seconds: int = 0,
    custom_attrs: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Send a message to an SQS queue.

    Args:
        message: Message payload (will be JSON-serialized)
        queue_name: Target SQS queue name
        message_type: Optional message type attribute
        job_id: Optional job ID for tracing
        priority: Optional priority level
        delay_seconds: Delay before message becomes visible (0-900s)
        custom_attrs: Additional message attributes

    Returns:
        Dict with 'message_id', 'sequence_number' (if FIFO), and 'md5_of_body'

    Raises:
        Exception: If send fails
    """
    try:
        queue_url = _get_queue_url(queue_name)

        attrs = _build_message_attributes(
            message_type=message_type,
            job_id=job_id,
            priority=priority,
            custom_attrs=custom_attrs
        )

        kwargs: Dict[str, Any] = {
            'QueueUrl': queue_url,
            'MessageBody': json.dumps(message, default=str),
            'MessageAttributes': attrs,
        }

        if delay_seconds > 0:
            kwargs['DelaySeconds'] = min(delay_seconds, 900)

        response = clients.sqs.send_message(**kwargs)

        result = {
            'message_id': response.get('MessageId'),
            'md5_of_body': response.get('MD5OfMessageBody'),
        }

        if 'SequenceNumber' in response:
            result['sequence_number'] = response['SequenceNumber']

        logger.info(
            "Sent message to %s | message_id=%s | job_id=%s",
            queue_name, result['message_id'], job_id
        )

        return result

    except ClientError as e:
        logger.error("Failed to send message to %s: %s", queue_name, e)
        raise Exception(f"Failed to send SQS message: {e}")


def send_generation_request(
    job_id: str,
    request_data: Dict[str, Any],
    priority: str = "normal",
    delay_seconds: int = 0
) -> Dict[str, Any]:
    """
    Send a generation request to the generation queue.

    Args:
        job_id: Unique job identifier
        request_data: Preprocessed request payload
        priority: Priority level ('high', 'normal', 'low')
        delay_seconds: Optional delay

    Returns:
        Send result with message_id
    """
    message = {
        "job_id": job_id,
        "request_data": request_data,
        "submitted_at": int(time.time()),
        "queue": "generation"
    }

    return send_message(
        message=message,
        queue_name=SQS_GENERATION_QUEUE,
        message_type="GenerationRequest",
        job_id=job_id,
        priority=priority,
        delay_seconds=delay_seconds
    )


def send_postprocessing_request(
    job_id: str,
    generation_result: Dict[str, Any],
    request_data: Dict[str, Any],
    priority: str = "normal"
) -> Dict[str, Any]:
    """
    Send a postprocessing request to the postprocessing queue.

    Args:
        job_id: Unique job identifier
        generation_result: Output from generation Lambda
        request_data: Original request data
        priority: Priority level

    Returns:
        Send result with message_id
    """
    message = {
        "job_id": job_id,
        "generation_result": generation_result,
        "request_data": request_data,
        "submitted_at": int(time.time()),
        "queue": "postprocessing"
    }

    return send_message(
        message=message,
        queue_name=SQS_POSTPROCESSING_QUEUE,
        message_type="PostprocessingRequest",
        job_id=job_id,
        priority=priority
    )


def send_notification_request(
    job_id: str,
    notification_type: str,
    notification_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send a notification request to the notification queue.

    Args:
        job_id: Job identifier
        notification_type: Type of notification ('success', 'failure', 'alert')
        notification_data: Notification payload

    Returns:
        Send result with message_id
    """
    message = {
        "job_id": job_id,
        "notification_type": notification_type,
        "notification_data": notification_data,
        "submitted_at": int(time.time()),
        "queue": "notification"
    }

    return send_message(
        message=message,
        queue_name=SQS_NOTIFICATION_QUEUE,
        message_type="NotificationRequest",
        job_id=job_id,
        priority="high" if notification_type == "failure" else "normal"
    )


def send_batch_messages(
    messages: List[Dict[str, Any]],
    queue_name: str = SQS_GENERATION_QUEUE,
    message_type: str = "GenerationRequest"
) -> Dict[str, Any]:
    """
    Send multiple messages to an SQS queue using batch operations.

    Args:
        messages: List of message payloads
        queue_name: Target SQS queue name
        message_type: Message type attribute for all messages

    Returns:
        Dict with 'successful' and 'failed' lists
    """
    queue_url = _get_queue_url(queue_name)
    successful = []
    failed = []

    # Process in batches of 10 (SQS limit)
    for i in range(0, len(messages), MAX_SQS_BATCH_SIZE):
        batch = messages[i:i + MAX_SQS_BATCH_SIZE]

        entries = []
        for idx, msg in enumerate(batch):
            job_id = msg.get("job_id", f"batch-{i + idx}")
            entry = {
                'Id': f"msg-{i + idx}-{job_id}",
                'MessageBody': json.dumps(msg, default=str),
                'MessageAttributes': _build_message_attributes(
                    message_type=message_type,
                    job_id=job_id
                )
            }
            entries.append(entry)

        try:
            response = clients.sqs.send_message_batch(
                QueueUrl=queue_url,
                Entries=entries
            )

            for success in response.get('Successful', []):
                successful.append({
                    'id': success['Id'],
                    'message_id': success.get('MessageId')
                })

            for failure in response.get('Failed', []):
                failed.append({
                    'id': failure['Id'],
                    'code': failure.get('Code'),
                    'message': failure.get('Message'),
                    'sender_fault': failure.get('SenderFault', False)
                })

        except ClientError as e:
            logger.error("Batch send failed for batch starting at %d: %s", i, e)
            for idx in range(len(batch)):
                failed.append({
                    'id': f"msg-{i + idx}",
                    'code': 'ClientError',
                    'message': str(e),
                    'sender_fault': False
                })

    logger.info(
        "Batch send to %s: %d successful, %d failed",
        queue_name, len(successful), len(failed)
    )

    return {'successful': successful, 'failed': failed}


# ---------------------------------------------------------------------------
# Receive Operations
# ---------------------------------------------------------------------------

def receive_messages(
    queue_name: str = SQS_GENERATION_QUEUE,
    max_messages: int = MAX_SQS_BATCH_SIZE,
    visibility_timeout: int = DEFAULT_VISIBILITY_TIMEOUT,
    wait_time_seconds: int = 20,
    attribute_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Receive messages from an SQS queue using long polling.

    Args:
        queue_name: Source SQS queue name
        max_messages: Maximum number of messages to retrieve (1-10)
        visibility_timeout: Visibility timeout in seconds
        wait_time_seconds: Long poll wait time (0-20s)
        attribute_names: Optional list of attribute names to retrieve

    Returns:
        List of parsed message dictionaries with metadata
    """
    try:
        queue_url = _get_queue_url(queue_name)

        kwargs: Dict[str, Any] = {
            'QueueUrl': queue_url,
            'MaxNumberOfMessages': min(max_messages, MAX_SQS_BATCH_SIZE),
            'VisibilityTimeout': visibility_timeout,
            'WaitTimeSeconds': min(wait_time_seconds, 20),
            'AttributeNames': attribute_names or ['All'],
            'MessageAttributeNames': ['All']
        }

        response = clients.sqs.receive_message(**kwargs)

        messages = []
        for msg in response.get('Messages', []):
            parsed = {
                'message_id': msg.get('MessageId'),
                'receipt_handle': msg.get('ReceiptHandle'),
                'body': json.loads(msg.get('Body', '{}')),
                'attributes': msg.get('Attributes', {}),
                'message_attributes': msg.get('MessageAttributes', {}),
                'md5_of_body': msg.get('MD5OfBody')
            }
            messages.append(parsed)

        if messages:
            logger.info("Received %d messages from %s", len(messages), queue_name)

        return messages

    except ClientError as e:
        logger.error("Failed to receive messages from %s: %s", queue_name, e)
        raise Exception(f"Failed to receive SQS messages: {e}")


def receive_generation_messages(
    max_messages: int = MAX_SQS_BATCH_SIZE,
    visibility_timeout: int = DEFAULT_VISIBILITY_TIMEOUT
) -> List[Dict[str, Any]]:
    """
    Receive messages from the generation queue.

    Args:
        max_messages: Maximum messages to retrieve
        visibility_timeout: Visibility timeout in seconds

    Returns:
        List of generation request messages
    """
    return receive_messages(
        queue_name=SQS_GENERATION_QUEUE,
        max_messages=max_messages,
        visibility_timeout=visibility_timeout
    )


def receive_postprocessing_messages(
    max_messages: int = MAX_SQS_BATCH_SIZE,
    visibility_timeout: int = DEFAULT_VISIBILITY_TIMEOUT
) -> List[Dict[str, Any]]:
    """
    Receive messages from the postprocessing queue.

    Args:
        max_messages: Maximum messages to retrieve
        visibility_timeout: Visibility timeout in seconds

    Returns:
        List of postprocessing request messages
    """
    return receive_messages(
        queue_name=SQS_POSTPROCESSING_QUEUE,
        max_messages=max_messages,
        visibility_timeout=visibility_timeout
    )


def receive_notification_messages(
    max_messages: int = MAX_SQS_BATCH_SIZE,
    visibility_timeout: int = 60
) -> List[Dict[str, Any]]:
    """
    Receive messages from the notification queue.

    Args:
        max_messages: Maximum messages to retrieve
        visibility_timeout: Visibility timeout in seconds

    Returns:
        List of notification request messages
    """
    return receive_messages(
        queue_name=SQS_NOTIFICATION_QUEUE,
        max_messages=max_messages,
        visibility_timeout=visibility_timeout
    )


# ---------------------------------------------------------------------------
# Message Lifecycle Operations
# ---------------------------------------------------------------------------

def delete_message(
    receipt_handle: str,
    queue_name: str = SQS_GENERATION_QUEUE
) -> bool:
    """
    Delete a message from the queue after successful processing.

    Args:
        receipt_handle: Message receipt handle
        queue_name: Source SQS queue name

    Returns:
        True if deletion succeeded
    """
    try:
        queue_url = _get_queue_url(queue_name)
        clients.sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        logger.info("Deleted message from %s", queue_name)
        return True
    except ClientError as e:
        logger.error("Failed to delete message from %s: %s", queue_name, e)
        return False


def delete_message_batch(
    receipt_handles: List[Dict[str, str]],
    queue_name: str = SQS_GENERATION_QUEUE
) -> Dict[str, Any]:
    """
    Delete multiple messages from the queue in a batch.

    Args:
        receipt_handles: List of dicts with 'id' and 'receipt_handle' keys
        queue_name: Source SQS queue name

    Returns:
        Dict with 'successful' and 'failed' lists
    """
    if not receipt_handles:
        return {'successful': [], 'failed': []}

    queue_url = _get_queue_url(queue_name)
    successful = []
    failed = []

    for i in range(0, len(receipt_handles), MAX_SQS_BATCH_SIZE):
        batch = receipt_handles[i:i + MAX_SQS_BATCH_SIZE]

        entries = [
            {
                'Id': item.get('id', f"del-{i + idx}"),
                'ReceiptHandle': item['receipt_handle']
            }
            for idx, item in enumerate(batch)
        ]

        try:
            response = clients.sqs.delete_message_batch(
                QueueUrl=queue_url,
                Entries=entries
            )

            successful.extend([s['Id'] for s in response.get('Successful', [])])
            failed.extend(response.get('Failed', []))

        except ClientError as e:
            logger.error("Batch delete failed: %s", e)
            failed.extend([
                {'Id': entry['Id'], 'Code': 'ClientError', 'Message': str(e)}
                for entry in entries
            ])

    logger.info("Batch delete from %s: %d ok, %d failed", queue_name, len(successful), len(failed))
    return {'successful': successful, 'failed': failed}


def change_visibility(
    receipt_handle: str,
    visibility_timeout: int,
    queue_name: str = SQS_GENERATION_QUEUE
) -> bool:
    """
    Change the visibility timeout of a message.

    Useful for extending processing time or making messages immediately
    visible again (set timeout to 0).

    Args:
        receipt_handle: Message receipt handle
        visibility_timeout: New visibility timeout in seconds
        queue_name: Source SQS queue name

    Returns:
        True if change succeeded
    """
    try:
        queue_url = _get_queue_url(queue_name)
        clients.sqs.change_message_visibility(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout
        )
        logger.info("Changed visibility to %ds for message in %s", visibility_timeout, queue_name)
        return True
    except ClientError as e:
        logger.error("Failed to change message visibility: %s", e)
        return False


def return_message_to_queue(
    receipt_handle: str,
    queue_name: str = SQS_GENERATION_QUEUE
) -> bool:
    """
    Return a message to the queue immediately by setting visibility to 0.

    Args:
        receipt_handle: Message receipt handle
        queue_name: Source SQS queue name

    Returns:
        True if operation succeeded
    """
    return change_visibility(receipt_handle, 0, queue_name)


# ---------------------------------------------------------------------------
# DLQ Operations
# ---------------------------------------------------------------------------

def get_dlq_message_count(queue_name: str = SQS_DLQ) -> Optional[int]:
    """
    Get the approximate number of messages in the dead-letter queue.

    Args:
        queue_name: DLQ queue name

    Returns:
        Approximate message count or None if unavailable
    """
    try:
        queue_url = _get_queue_url(queue_name)
        response = clients.sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        return int(response.get('Attributes', {}).get('ApproximateNumberOfMessages', 0))
    except ClientError as e:
        logger.error("Failed to get DLQ message count: %s", e)
        return None


def receive_dlq_messages(
    max_messages: int = MAX_SQS_BATCH_SIZE,
    visibility_timeout: int = 60
) -> List[Dict[str, Any]]:
    """
    Receive messages from the dead-letter queue for inspection/replay.

    Args:
        max_messages: Maximum messages to retrieve
        visibility_timeout: Visibility timeout in seconds

    Returns:
        List of DLQ messages
    """
    return receive_messages(
        queue_name=SQS_DLQ,
        max_messages=max_messages,
        visibility_timeout=visibility_timeout,
        wait_time_seconds=0  # No long polling for DLQ
    )


def replay_dlq_message(
    receipt_handle: str,
    message_body: Dict[str, Any],
    target_queue: str = SQS_GENERATION_QUEUE,
    dlq_name: str = SQS_DLQ
) -> Dict[str, Any]:
    """
    Replay a message from the DLQ back to the main queue.

    Sends the message to the target queue and deletes it from the DLQ.

    Args:
        receipt_handle: DLQ message receipt handle
        message_body: Original message body to replay
        target_queue: Target queue to send the message to
        dlq_name: DLQ queue name

    Returns:
        Send result from the target queue
    """
    # Send to target queue
    result = send_message(
        message=message_body,
        queue_name=target_queue,
        message_type="ReplayRequest"
    )

    # Delete from DLQ
    delete_message(receipt_handle, dlq_name)

    logger.info("Replayed DLQ message to %s", target_queue)
    return result


def replay_all_dlq_messages(
    target_queue: str = SQS_GENERATION_QUEUE
) -> Dict[str, Any]:
    """
    Replay all messages from the DLQ back to the main queue.

    Args:
        target_queue: Target queue for replayed messages

    Returns:
        Dict with 'replayed' count and 'errors' list
    """
    replayed = 0
    errors = []

    while True:
        messages = receive_dlq_messages()
        if not messages:
            break

        for msg in messages:
            try:
                replay_dlq_message(
                    receipt_handle=msg['receipt_handle'],
                    message_body=msg['body'],
                    target_queue=target_queue
                )
                replayed += 1
            except Exception as e:
                errors.append({
                    'message_id': msg.get('message_id'),
                    'error': str(e)
                })

    logger.info("Replayed %d DLQ messages, %d errors", replayed, len(errors))
    return {'replayed': replayed, 'errors': errors}


# ---------------------------------------------------------------------------
# Queue Monitoring
# ---------------------------------------------------------------------------

def get_queue_attributes(queue_name: str = SQS_GENERATION_QUEUE) -> Dict[str, Any]:
    """
    Get comprehensive queue attributes and metrics.

    Args:
        queue_name: SQS queue name

    Returns:
        Dict with queue attributes
    """
    try:
        queue_url = _get_queue_url(queue_name)
        response = clients.sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=[
                'ApproximateNumberOfMessages',
                'ApproximateNumberOfMessagesNotVisible',
                'ApproximateNumberOfMessagesDelayed',
                'CreatedTimestamp',
                'LastModifiedTimestamp',
                'VisibilityTimeout',
                'MaximumMessageSize',
                'MessageRetentionPeriod',
                'DelaySeconds',
                'ReceiveMessageWaitTimeSeconds',
                'RedrivePolicy',
                'QueueArn'
            ]
        )

        attrs = response.get('Attributes', {})

        return {
            'queue_name': queue_name,
            'queue_url': queue_url,
            'approximate_messages': int(attrs.get('ApproximateNumberOfMessages', 0)),
            'messages_in_flight': int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0)),
            'messages_delayed': int(attrs.get('ApproximateNumberOfMessagesDelayed', 0)),
            'visibility_timeout': int(attrs.get('VisibilityTimeout', 0)),
            'maximum_message_size': int(attrs.get('MaximumMessageSize', 0)),
            'message_retention_period': int(attrs.get('MessageRetentionPeriod', 0)),
            'delay_seconds': int(attrs.get('DelaySeconds', 0)),
            'wait_time_seconds': int(attrs.get('ReceiveMessageWaitTimeSeconds', 0)),
            'queue_arn': attrs.get('QueueArn', ''),
            'redrive_policy': attrs.get('RedrivePolicy', ''),
            'created_timestamp': attrs.get('CreatedTimestamp', ''),
            'last_modified_timestamp': attrs.get('LastModifiedTimestamp', '')
        }

    except ClientError as e:
        logger.error("Failed to get queue attributes for %s: %s", queue_name, e)
        raise Exception(f"Failed to get queue attributes: {e}")


def get_all_queue_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get metrics for all pipeline queues.

    Returns:
        Dict mapping queue names to their attributes
    """
    queues = [
        SQS_GENERATION_QUEUE,
        SQS_POSTPROCESSING_QUEUE,
        SQS_NOTIFICATION_QUEUE,
        SQS_DLQ
    ]

    metrics = {}
    for queue_name in queues:
        try:
            metrics[queue_name] = get_queue_attributes(queue_name)
        except Exception as e:
            logger.warning("Failed to get metrics for %s: %s", queue_name, e)
            metrics[queue_name] = {'error': str(e)}

    return metrics


def purge_queue(queue_name: str) -> bool:
    """
    Purge all messages from a queue. Use with extreme caution!

    Args:
        queue_name: SQS queue name to purge

    Returns:
        True if purge succeeded
    """
    try:
        queue_url = _get_queue_url(queue_name)
        clients.sqs.purge_queue(QueueUrl=queue_url)
        logger.warning("Purged queue: %s", queue_name)
        return True
    except ClientError as e:
        logger.error("Failed to purge queue %s: %s", queue_name, e)
        return False


def get_sqs_client():
    """Get the underlying SQS client for advanced operations."""
    return clients.sqs

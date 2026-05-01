"""
SNS Notification Manager
------------------------
Comprehensive SNS operations for the Anime Quote Generator pipeline:
1. Topic-based notification publishing
2. Event-type routing (success, failure, alert, system)
3. Subscription management (email, Lambda, SQS, HTTPS)
4. Notification templating and formatting
5. Alert escalation for critical failures
6. Batch notification delivery

SNS Topic Architecture:
    generation-complete    → Notifications for successful generations
    generation-failed      → Notifications for failed generations
    system-alerts          → System-level alerts (DLQ overflow, throttling, errors)
    user-notifications     → User-facing notification delivery
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
    SNS_GENERATION_COMPLETE, SNS_GENERATION_FAILED,
    SNS_SYSTEM_ALERTS, SNS_USER_NOTIFICATIONS,
    EVENT_GENERATION_COMPLETED, EVENT_GENERATION_FAILED,
    EVENT_PREPROCESSING_FAILED, EVENT_POSTPROCESSING_FAILED,
    EVENT_DLQ_OVERFLOW, EVENT_RATE_LIMIT_EXCEEDED,
    EVENT_SYSTEM_ERROR, EVENT_MODEL_UNAVAILABLE,
    SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR, SEVERITY_CRITICAL
)

logger = logging.getLogger(__name__)


def _get_topic_arn(topic_name: str) -> str:
    """
    Resolve an SNS topic name to its ARN.

    Checks environment variables first, then constructs ARN from
    AWS region and account ID.

    Args:
        topic_name: SNS topic name

    Returns:
        SNS topic ARN string
    """
    # Check environment variable first (set by CDK/CloudFormation)
    env_var = f"SNS_{topic_name.upper().replace('-', '_')}"
    env_arn = os.environ.get(env_var)
    if env_arn:
        return env_arn

    # Construct ARN from environment
    region = os.environ.get('AWS_REGION', 'us-east-1')
    account_id = os.environ.get('AWS_ACCOUNT_ID', '123456789012')
    return f"arn:aws:sns:{region}:{account_id}:{topic_name}"


def _build_message_attributes(
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    job_id: Optional[str] = None,
    source: Optional[str] = None,
    custom_attrs: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Dict[str, str]]:
    """
    Build SNS message attributes for filtering and routing.

    Args:
        event_type: Event type identifier
        severity: Severity level (info, warning, error, critical)
        job_id: Associated job ID
        source: Source service/function name
        custom_attrs: Additional custom attributes

    Returns:
        SNS message attributes dictionary
    """
    attrs: Dict[str, Dict[str, str]] = {}

    if event_type:
        attrs['EventType'] = {
            'DataType': 'String',
            'StringValue': event_type
        }

    if severity:
        attrs['Severity'] = {
            'DataType': 'String',
            'StringValue': severity
        }

    if job_id:
        attrs['JobId'] = {
            'DataType': 'String',
            'StringValue': job_id
        }

    if source:
        attrs['Source'] = {
            'DataType': 'String',
            'StringValue': source
        }

    attrs['Timestamp'] = {
        'DataType': 'String',
        'StringValue': datetime.utcnow().isoformat() + "Z"
    }

    if custom_attrs:
        attrs.update(custom_attrs)

    return attrs


# ---------------------------------------------------------------------------
# Publish Operations
# ---------------------------------------------------------------------------

def publish_notification(
    topic_name: str,
    subject: str,
    message: Dict[str, Any],
    event_type: Optional[str] = None,
    severity: str = SEVERITY_INFO,
    job_id: Optional[str] = None,
    source: Optional[str] = None,
    custom_attrs: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Publish a notification to an SNS topic.

    Args:
        topic_name: SNS topic name
        subject: Message subject (max 100 chars)
        message: Message payload (will be JSON-serialized)
        event_type: Event type for filtering
        severity: Severity level
        job_id: Associated job ID
        source: Source service/function
        custom_attrs: Additional message attributes

    Returns:
        Dict with 'message_id' and 'sequence_number' (if FIFO)

    Raises:
        Exception: If publish fails
    """
    try:
        topic_arn = _get_topic_arn(topic_name)

        attrs = _build_message_attributes(
            event_type=event_type,
            severity=severity,
            job_id=job_id,
            source=source,
            custom_attrs=custom_attrs
        )

        # Truncate subject to 100 chars (SNS limit)
        truncated_subject = subject[:100] if len(subject) > 100 else subject

        kwargs: Dict[str, Any] = {
            'TopicArn': topic_arn,
            'Subject': truncated_subject,
            'Message': json.dumps(message, default=str, indent=2),
            'MessageAttributes': attrs
        }

        response = clients.sns.publish(**kwargs)

        result = {
            'message_id': response.get('MessageId'),
            'topic_arn': topic_arn
        }

        if 'SequenceNumber' in response:
            result['sequence_number'] = response['SequenceNumber']

        logger.info(
            "Published to %s | message_id=%s | event=%s | severity=%s",
            topic_name, result['message_id'], event_type, severity
        )

        return result

    except ClientError as e:
        logger.error("Failed to publish to SNS topic %s: %s", topic_name, e)
        raise Exception(f"Failed to publish SNS notification: {e}")


def publish_generation_success(
    job_id: str,
    result_summary: Dict[str, Any],
    source: str = "postprocessing"
) -> Dict[str, Any]:
    """
    Publish a generation success notification.

    Args:
        job_id: Completed job identifier
        result_summary: Summary of the generation result
        source: Source function name

    Returns:
        Publish result with message_id
    """
    notification = {
        "event": EVENT_GENERATION_COMPLETED,
        "job_id": job_id,
        "timestamp": int(time.time()),
        "result": result_summary,
        "message": f"Generation completed successfully for job {job_id}"
    }

    return publish_notification(
        topic_name=SNS_GENERATION_COMPLETE,
        subject=f"Generation Completed: {job_id}",
        message=notification,
        event_type=EVENT_GENERATION_COMPLETED,
        severity=SEVERITY_INFO,
        job_id=job_id,
        source=source
    )


def publish_generation_failure(
    job_id: str,
    error: str,
    stage: str = "unknown",
    retry_count: int = 0,
    source: str = "generation"
) -> Dict[str, Any]:
    """
    Publish a generation failure notification.

    Args:
        job_id: Failed job identifier
        error: Error message
        stage: Pipeline stage where failure occurred
        retry_count: Number of retry attempts
        source: Source function name

    Returns:
        Publish result with message_id
    """
    severity = SEVERITY_CRITICAL if retry_count >= 3 else SEVERITY_ERROR

    notification = {
        "event": EVENT_GENERATION_FAILED,
        "job_id": job_id,
        "timestamp": int(time.time()),
        "error": error,
        "stage": stage,
        "retry_count": retry_count,
        "message": f"Generation failed for job {job_id} at stage '{stage}': {error}"
    }

    return publish_notification(
        topic_name=SNS_GENERATION_FAILED,
        subject=f"Generation Failed: {job_id} ({stage})",
        message=notification,
        event_type=EVENT_GENERATION_FAILED,
        severity=severity,
        job_id=job_id,
        source=source
    )


def publish_system_alert(
    alert_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    severity: str = SEVERITY_WARNING
) -> Dict[str, Any]:
    """
    Publish a system-level alert notification.

    Args:
        alert_type: Type of alert (e.g., 'dlq_overflow', 'rate_limit', 'model_unavailable')
        message: Alert message
        details: Additional alert details
        severity: Alert severity level

    Returns:
        Publish result with message_id
    """
    notification = {
        "event": alert_type,
        "timestamp": int(time.time()),
        "message": message,
        "details": details or {},
        "severity": severity
    }

    return publish_notification(
        topic_name=SNS_SYSTEM_ALERTS,
        subject=f"System Alert: {alert_type}",
        message=notification,
        event_type=alert_type,
        severity=severity,
        source="system"
    )


def publish_user_notification(
    user_id: str,
    notification_type: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Publish a user-facing notification.

    Args:
        user_id: Target user identifier
        notification_type: Type of notification ('generation_ready', 'batch_complete', etc.)
        title: Notification title
        body: Notification body text
        data: Additional notification data

    Returns:
        Publish result with message_id
    """
    notification = {
        "event": notification_type,
        "user_id": user_id,
        "timestamp": int(time.time()),
        "title": title,
        "body": body,
        "data": data or {}
    }

    return publish_notification(
        topic_name=SNS_USER_NOTIFICATIONS,
        subject=title[:100],
        message=notification,
        event_type=notification_type,
        severity=SEVERITY_INFO,
        custom_attrs={'UserId': {'DataType': 'String', 'StringValue': user_id}}
    )


def publish_dlq_alert(
    queue_name: str,
    message_count: int,
    threshold: int = 10
) -> Dict[str, Any]:
    """
    Publish an alert when the DLQ exceeds a message threshold.

    Args:
        queue_name: DLQ queue name
        message_count: Current message count in DLQ
        threshold: Alert threshold

    Returns:
        Publish result with message_id
    """
    severity = SEVERITY_CRITICAL if message_count > threshold * 2 else SEVERITY_ERROR

    return publish_system_alert(
        alert_type=EVENT_DLQ_OVERFLOW,
        message=f"DLQ '{queue_name}' has {message_count} messages (threshold: {threshold})",
        details={
            "queue_name": queue_name,
            "message_count": message_count,
            "threshold": threshold,
            "overflow_ratio": round(message_count / threshold, 2) if threshold > 0 else 0
        },
        severity=severity
    )


def publish_model_unavailable_alert(
    model_name: str,
    fallback_used: str,
    error: str
) -> Dict[str, Any]:
    """
    Publish an alert when a model is unavailable and fallback was used.

    Args:
        model_name: Name of the unavailable model
        fallback_used: Name of the fallback model used
        error: Error message from the model failure

    Returns:
        Publish result with message_id
    """
    return publish_system_alert(
        alert_type=EVENT_MODEL_UNAVAILABLE,
        message=f"Model '{model_name}' unavailable, using fallback '{fallback_used}'",
        details={
            "model_name": model_name,
            "fallback_used": fallback_used,
            "error": error
        },
        severity=SEVERITY_WARNING
    )


# ---------------------------------------------------------------------------
# Batch Publish Operations
# ---------------------------------------------------------------------------

def publish_batch_notifications(
    notifications: List[Dict[str, Any]],
    topic_name: str = SNS_GENERATION_COMPLETE
) -> Dict[str, Any]:
    """
    Publish multiple notifications to the same SNS topic.

    Args:
        notifications: List of notification dicts, each with 'subject', 'message',
            and optional 'event_type', 'severity', 'job_id'
        topic_name: Target SNS topic name

    Returns:
        Dict with 'successful' and 'failed' counts
    """
    successful = 0
    failed = []

    for notif in notifications:
        try:
            publish_notification(
                topic_name=topic_name,
                subject=notif.get('subject', 'Batch Notification'),
                message=notif.get('message', {}),
                event_type=notif.get('event_type'),
                severity=notif.get('severity', SEVERITY_INFO),
                job_id=notif.get('job_id'),
                source=notif.get('source', 'batch')
            )
            successful += 1
        except Exception as e:
            failed.append({
                'subject': notif.get('subject'),
                'error': str(e)
            })

    logger.info("Batch publish to %s: %d ok, %d failed", topic_name, successful, len(failed))
    return {'successful': successful, 'failed': failed}


# ---------------------------------------------------------------------------
# Subscription Management
# ---------------------------------------------------------------------------

def subscribe_email(
    topic_name: str,
    email_address: str
) -> Optional[str]:
    """
    Subscribe an email address to an SNS topic.

    Args:
        topic_name: SNS topic name
        email_address: Email address to subscribe

    Returns:
        Subscription ARN or None if failed
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        response = clients.sns.subscribe(
            TopicArn=topic_arn,
            Protocol='email',
            Endpoint=email_address
        )
        sub_arn = response.get('SubscriptionArn', 'pending confirmation')
        logger.info("Subscribed %s to %s (ARN: %s)", email_address, topic_name, sub_arn)
        return sub_arn
    except ClientError as e:
        logger.error("Failed to subscribe email %s: %s", email_address, e)
        return None


def subscribe_lambda(
    topic_name: str,
    function_arn: str
) -> Optional[str]:
    """
    Subscribe a Lambda function to an SNS topic.

    Args:
        topic_name: SNS topic name
        function_arn: Lambda function ARN

    Returns:
        Subscription ARN or None if failed
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        response = clients.sns.subscribe(
            TopicArn=topic_arn,
            Protocol='lambda',
            Endpoint=function_arn
        )
        sub_arn = response.get('SubscriptionArn')
        logger.info("Subscribed Lambda %s to %s", function_arn, topic_name)
        return sub_arn
    except ClientError as e:
        logger.error("Failed to subscribe Lambda %s: %s", function_arn, e)
        return None


def subscribe_sqs_queue(
    topic_name: str,
    queue_arn: str
) -> Optional[str]:
    """
    Subscribe an SQS queue to an SNS topic.

    Args:
        topic_name: SNS topic name
        queue_arn: SQS queue ARN

    Returns:
        Subscription ARN or None if failed
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        response = clients.sns.subscribe(
            TopicArn=topic_arn,
            Protocol='sqs',
            Endpoint=queue_arn
        )
        sub_arn = response.get('SubscriptionArn')
        logger.info("Subscribed SQS %s to %s", queue_arn, topic_name)
        return sub_arn
    except ClientError as e:
        logger.error("Failed to subscribe SQS %s: %s", queue_arn, e)
        return None


def subscribe_https(
    topic_name: str,
    endpoint_url: str
) -> Optional[str]:
    """
    Subscribe an HTTPS endpoint to an SNS topic.

    Args:
        topic_name: SNS topic name
        endpoint_url: HTTPS endpoint URL

    Returns:
        Subscription ARN or None if failed
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        response = clients.sns.subscribe(
            TopicArn=topic_arn,
            Protocol='https',
            Endpoint=endpoint_url
        )
        sub_arn = response.get('SubscriptionArn')
        logger.info("Subscribed HTTPS %s to %s", endpoint_url, topic_name)
        return sub_arn
    except ClientError as e:
        logger.error("Failed to subscribe HTTPS %s: %s", endpoint_url, e)
        return None


def unsubscribe(subscription_arn: str) -> bool:
    """
    Unsubscribe from an SNS topic.

    Args:
        subscription_arn: Subscription ARN to remove

    Returns:
        True if unsubscription succeeded
    """
    try:
        clients.sns.unsubscribe(SubscriptionArn=subscription_arn)
        logger.info("Unsubscribed %s", subscription_arn)
        return True
    except ClientError as e:
        logger.error("Failed to unsubscribe %s: %s", subscription_arn, e)
        return False


def list_subscriptions(topic_name: str) -> List[Dict[str, Any]]:
    """
    List all subscriptions for an SNS topic.

    Args:
        topic_name: SNS topic name

    Returns:
        List of subscription details
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        response = clients.sns.list_subscriptions_by_topic(TopicArn=topic_arn)

        subscriptions = []
        for sub in response.get('Subscriptions', []):
            subscriptions.append({
                'subscription_arn': sub.get('SubscriptionArn'),
                'protocol': sub.get('Protocol'),
                'endpoint': sub.get('Endpoint'),
                'owner': sub.get('Owner')
            })

        return subscriptions

    except ClientError as e:
        logger.error("Failed to list subscriptions for %s: %s", topic_name, e)
        return []


# ---------------------------------------------------------------------------
# Topic Management
# ---------------------------------------------------------------------------

def get_topic_attributes(topic_name: str) -> Dict[str, Any]:
    """
    Get attributes for an SNS topic.

    Args:
        topic_name: SNS topic name

    Returns:
        Dict with topic attributes
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        response = clients.sns.get_topic_attributes(TopicArn=topic_arn)
        attrs = response.get('Attributes', {})

        return {
            'topic_arn': attrs.get('TopicArn'),
            'display_name': attrs.get('DisplayName', ''),
            'subscriptions_confirmed': int(attrs.get('SubscriptionsConfirmed', 0)),
            'subscriptions_pending': int(attrs.get('SubscriptionsPending', 0)),
            'subscriptions_deleted': int(attrs.get('SubscriptionsDeleted', 0))
        }

    except ClientError as e:
        logger.error("Failed to get topic attributes for %s: %s", topic_name, e)
        raise Exception(f"Failed to get SNS topic attributes: {e}")


def set_topic_display_name(topic_name: str, display_name: str) -> bool:
    """
    Set the display name for an SNS topic (used in email subjects).

    Args:
        topic_name: SNS topic name
        display_name: Display name (max 100 chars)

    Returns:
        True if succeeded
    """
    try:
        topic_arn = _get_topic_arn(topic_name)
        clients.sns.set_topic_attributes(
            TopicArn=topic_arn,
            AttributeName='DisplayName',
            AttributeValue=display_name[:100]
        )
        return True
    except ClientError as e:
        logger.error("Failed to set display name for %s: %s", topic_name, e)
        return False


# ---------------------------------------------------------------------------
# Notification Formatting
# ---------------------------------------------------------------------------

def format_generation_success_message(
    job_id: str,
    speech_type: str,
    generation_type: str,
    generation_method: str,
    word_count: int,
    s3_key: str,
    processing_time: float
) -> Dict[str, Any]:
    """
    Format a standardized generation success notification.

    Args:
        job_id: Job identifier
        speech_type: Type of speech generated
        generation_type: Type of generation (speech/dialogue)
        generation_method: Method used (gemini/gpt2/fallback)
        word_count: Word count of generated content
        s3_key: S3 key where output is stored
        processing_time: Total processing time in seconds

    Returns:
        Formatted notification message
    """
    return {
        "event": EVENT_GENERATION_COMPLETED,
        "job_id": job_id,
        "timestamp": int(time.time()),
        "result": {
            "speech_type": speech_type,
            "generation_type": generation_type,
            "generation_method": generation_method,
            "word_count": word_count,
            "s3_key": s3_key,
            "processing_time_seconds": round(processing_time, 2)
        },
        "message": (
            f"Successfully generated {generation_type} ({speech_type}) "
            f"using {generation_method} in {processing_time:.2f}s"
        )
    }


def format_generation_failure_message(
    job_id: str,
    stage: str,
    error: str,
    retry_count: int = 0,
    max_retries: int = 3,
    request_details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format a standardized generation failure notification.

    Args:
        job_id: Job identifier
        stage: Pipeline stage where failure occurred
        error: Error message
        retry_count: Current retry attempt
        max_retries: Maximum retry attempts
        request_details: Original request details for debugging

    Returns:
        Formatted notification message
    """
    is_terminal = retry_count >= max_retries

    return {
        "event": EVENT_GENERATION_FAILED,
        "job_id": job_id,
        "timestamp": int(time.time()),
        "failure": {
            "stage": stage,
            "error": error,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "is_terminal": is_terminal
        },
        "request_details": request_details or {},
        "message": (
            f"Generation failed at '{stage}' for job {job_id} "
            f"(retry {retry_count}/{max_retries}): {error}"
            + (" - TERMINAL FAILURE" if is_terminal else "")
        )
    }


def get_sns_client():
    """Get the underlying SNS client for advanced operations."""
    return clients.sns

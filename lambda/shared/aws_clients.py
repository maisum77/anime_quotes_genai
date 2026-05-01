"""
AWS client utilities for Lambda functions

This module provides the shared AWSClientManager singleton and backward-compatible
wrapper functions. For comprehensive operations, use the dedicated modules:
  - s3_storage:    Full S3 upload/download/lifecycle management
  - sqs_manager:   Full SQS send/receive/DLQ management
  - sns_manager:   Full SNS publish/subscribe management
  - dynamodb_manager: Full DynamoDB job/history/metrics management
"""
import os
import json
import boto3
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError
from .constants import (
    S3_BUCKET_PREFIX, S3_INPUTS_PREFIX, S3_OUTPUTS_PREFIX,
    S3_MODELS_PREFIX, S3_LOGS_PREFIX,
    DYNAMODB_JOBS_TABLE, DYNAMODB_HISTORY_TABLE, DYNAMODB_METRICS_TABLE,
    SQS_GENERATION_QUEUE, SQS_DLQ, SQS_NOTIFICATION_QUEUE,
    SNS_GENERATION_COMPLETE, SNS_GENERATION_FAILED
)

class AWSClientManager:
    """Manages AWS service clients with lazy initialization"""
    
    def __init__(self):
        self._s3 = None
        self._dynamodb = None
        self._sqs = None
        self._sns = None
        self._lambda_client = None
        self._cloudwatch = None
        
    @property
    def s3(self):
        if self._s3 is None:
            self._s3 = boto3.client('s3')
        return self._s3
    
    @property
    def dynamodb(self):
        if self._dynamodb is None:
            self._dynamodb = boto3.resource('dynamodb')
        return self._dynamodb
    
    @property
    def sqs(self):
        if self._sqs is None:
            self._sqs = boto3.client('sqs')
        return self._sqs
    
    @property
    def sns(self):
        if self._sns is None:
            self._sns = boto3.client('sns')
        return self._sns
    
    @property
    def lambda_client(self):
        if self._lambda_client is None:
            self._lambda_client = boto3.client('lambda')
        return self._lambda_client
    
    @property
    def cloudwatch(self):
        if self._cloudwatch is None:
            self._cloudwatch = boto3.client('cloudwatch')
        return self._cloudwatch

# Global client manager instance
clients = AWSClientManager()

def get_s3_bucket_name(environment: str = None) -> str:
    """
    Get S3 bucket name for environment
    
    Args:
        environment: Environment name (dev, staging, prod)
    
    Returns:
        S3 bucket name
    """
    if not environment:
        environment = os.environ.get('ENVIRONMENT', 'dev')
    
    return f"{S3_BUCKET_PREFIX}-{environment}"

def upload_to_s3(
    data: str,
    key: str,
    bucket: Optional[str] = None,
    content_type: str = "text/plain",
    metadata: Optional[Dict[str, str]] = None
) -> str:
    """
    Upload data to S3
    
    Args:
        data: Data to upload
        key: S3 object key
        bucket: S3 bucket name (uses default if None)
        content_type: Content type
        metadata: Optional metadata
    
    Returns:
        S3 object URL
    """
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    try:
        extra_args = {
            'ContentType': content_type,
        }
        
        if metadata:
            extra_args['Metadata'] = metadata
        
        clients.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            **extra_args
        )
        
        # Generate object URL
        location = clients.s3.get_bucket_location(Bucket=bucket)['LocationConstraint']
        if location is None:
            location = 'us-east-1'
        
        return f"https://{bucket}.s3.{location}.amazonaws.com/{key}"
        
    except ClientError as e:
        raise Exception(f"Failed to upload to S3: {e}")

def download_from_s3(key: str, bucket: Optional[str] = None) -> str:
    """
    Download data from S3
    
    Args:
        key: S3 object key
        bucket: S3 bucket name (uses default if None)
    
    Returns:
        Downloaded data as string
    """
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    try:
        response = clients.s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    except ClientError as e:
        raise Exception(f"Failed to download from S3: {e}")

def save_generation_result(
    job_id: str,
    result: Dict[str, Any],
    user_id: Optional[str] = None
) -> None:
    """
    Save generation result to DynamoDB
    
    Args:
        job_id: Job ID
        result: Generation result
        user_id: Optional user ID
    """
    try:
        table = clients.dynamodb.Table(DYNAMODB_JOBS_TABLE)
        
        item = {
            'job_id': job_id,
            'result': json.dumps(result),
            'timestamp': int(os.environ.get('AWS_LAMBDA_LOG_STREAM_NAME', '0').split('.')[0]),
            'status': 'completed'
        }
        
        if user_id:
            item['user_id'] = user_id
        
        table.put_item(Item=item)
        
    except ClientError as e:
        raise Exception(f"Failed to save to DynamoDB: {e}")

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job status from DynamoDB
    
    Args:
        job_id: Job ID
    
    Returns:
        Job status dictionary or None if not found
    """
    try:
        table = clients.dynamodb.Table(DYNAMODB_JOBS_TABLE)
        
        response = table.get_item(Key={'job_id': job_id})
        
        if 'Item' in response:
            item = response['Item']
            item['result'] = json.loads(item['result'])
            return item
        
        return None
        
    except ClientError as e:
        raise Exception(f"Failed to get job status: {e}")

def send_to_sqs(
    message: Dict[str, Any],
    queue_name: str = SQS_GENERATION_QUEUE
) -> str:
    """
    Send message to SQS queue
    
    Args:
        message: Message to send
        queue_name: Queue name
    
    Returns:
        Message ID
    """
    try:
        # Get queue URL
        response = clients.sqs.get_queue_url(QueueName=queue_name)
        queue_url = response['QueueUrl']
        
        # Send message
        response = clients.sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
            MessageAttributes={
                'MessageType': {
                    'DataType': 'String',
                    'StringValue': 'GenerationRequest'
                }
            }
        )
        
        return response['MessageId']
        
    except ClientError as e:
        raise Exception(f"Failed to send to SQS: {e}")

def publish_to_sns(
    subject: str,
    message: Dict[str, Any],
    topic_name: str = SNS_GENERATION_COMPLETE
) -> str:
    """
    Publish message to SNS topic
    
    Args:
        subject: Message subject
        message: Message to publish
        topic_name: Topic name
    
    Returns:
        Message ID
    """
    try:
        # Get topic ARN (in production, this would be from environment)
        topic_arn = f"arn:aws:sns:{os.environ.get('AWS_REGION', 'us-east-1')}:{os.environ.get('AWS_ACCOUNT_ID', '123456789012')}:{topic_name}"
        
        response = clients.sns.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=json.dumps(message),
            MessageAttributes={
                'EventType': {
                    'DataType': 'String',
                    'StringValue': 'GenerationEvent'
                }
            }
        )
        
        return response['MessageId']
        
    except ClientError as e:
        raise Exception(f"Failed to publish to SNS: {e}")

def invoke_lambda_async(
    function_name: str,
    payload: Dict[str, Any],
    invocation_type: str = 'Event'
) -> str:
    """
    Invoke Lambda function asynchronously
    
    Args:
        function_name: Lambda function name
        payload: Invocation payload
        invocation_type: Invocation type (Event for async)
    
    Returns:
        Request ID
    """
    try:
        response = clients.lambda_client.invoke(
            FunctionName=function_name,
            InvocationType=invocation_type,
            Payload=json.dumps(payload)
        )
        
        return response['ResponseMetadata']['RequestId']
        
    except ClientError as e:
        raise Exception(f"Failed to invoke Lambda: {e}")

def generate_s3_key(
    prefix: str,
    job_id: str,
    filename: str,
    timestamp: Optional[int] = None
) -> str:
    """
    Generate S3 key for storing generation results
    
    Args:
        prefix: S3 prefix (inputs, outputs, models, logs)
        job_id: Job ID
        filename: Filename
        timestamp: Optional timestamp
    
    Returns:
        S3 key
    """
    import time
    
    if timestamp is None:
        timestamp = int(time.time())
    
    date_str = time.strftime('%Y/%m/%d', time.gmtime(timestamp))
    
    return f"{prefix}/{date_str}/{job_id}/{filename}"

def list_s3_objects(
    prefix: str,
    bucket: Optional[str] = None,
    max_keys: int = 100
) -> List[Dict[str, Any]]:
    """
    List objects in S3 bucket with given prefix
    
    Args:
        prefix: S3 prefix
        bucket: S3 bucket name
        max_keys: Maximum number of keys to return
    
    Returns:
        List of object metadata
    """
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    try:
        response = clients.s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        if 'Contents' in response:
            return [
                {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag']
                }
                for obj in response['Contents']
            ]
        
        return []
        
    except ClientError as e:
        raise Exception(f"Failed to list S3 objects: {e}")

def delete_s3_object(key: str, bucket: Optional[str] = None) -> None:
    """
    Delete object from S3
    
    Args:
        key: S3 object key
        bucket: S3 bucket name
    """
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    try:
        clients.s3.delete_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise Exception(f"Failed to delete S3 object: {e}")


# ---------------------------------------------------------------------------
# Backward-Compatible Adapter Functions
# ---------------------------------------------------------------------------
# These functions provide the interface that Lambda functions expect,
# delegating to the dedicated manager modules. New code should import
# directly from the manager modules instead.
# ---------------------------------------------------------------------------

def get_dynamodb_client():
    """Get the DynamoDB resource. Prefer shared.dynamodb_manager for operations."""
    return clients.dynamodb


def get_sqs_client():
    """Get the SQS client. Prefer shared.sqs_manager for operations."""
    return clients.sqs


def get_s3_client():
    """Get the S3 client. Prefer shared.s3_storage for operations."""
    return clients.s3


def get_sns_client():
    """Get the SNS client. Prefer shared.sns_manager for operations."""
    return clients.sns


def update_job_status(job_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Update job status in DynamoDB.

    Delegates to shared.dynamodb_manager.update_job_status.
    """
    from .dynamodb_manager import update_job_status as _update_job_status
    return _update_job_status(job_id=job_id, status=status, metadata=metadata)


def store_user_history(history_item: Dict[str, Any]) -> bool:
    """
    Store a generation record in user history.

    Delegates to shared.dynamodb_manager.store_user_history.
    """
    from .dynamodb_manager import store_user_history as _store_user_history
    return _store_user_history(history_item=history_item)


def send_sqs_message(
    queue_url: str,
    message_body: str,
    message_attributes: Optional[Dict[str, Dict[str, str]]] = None,
    delay_seconds: int = 0
) -> str:
    """
    Send a message to an SQS queue by URL.

    Adapter that bridges the call signature used by Lambda functions
    to the shared.sqs_manager.send_message interface.

    Args:
        queue_url: SQS queue URL
        message_body: JSON-serialized message body
        message_attributes: Optional message attributes
        delay_seconds: Optional delay

    Returns:
        Message ID string
    """
    try:
        kwargs: Dict[str, Any] = {
            'QueueUrl': queue_url,
            'MessageBody': message_body,
        }
        if message_attributes:
            kwargs['MessageAttributes'] = message_attributes
        if delay_seconds > 0:
            kwargs['DelaySeconds'] = min(delay_seconds, 900)

        response = clients.sqs.send_message(**kwargs)
        return response.get('MessageId', '')

    except ClientError as e:
        raise Exception(f"Failed to send SQS message: {e}")


def publish_sns_notification(
    topic_arn: str,
    subject: str,
    message: str,
    message_attributes: Optional[Dict[str, Dict[str, str]]] = None
) -> str:
    """
    Publish a notification to an SNS topic by ARN.

    Adapter that bridges the call signature used by Lambda functions
    to the shared.sns_manager.publish_notification interface.

    Args:
        topic_arn: SNS topic ARN
        subject: Message subject
        message: Message body (JSON string)
        message_attributes: Optional message attributes

    Returns:
        Message ID string
    """
    try:
        kwargs: Dict[str, Any] = {
            'TopicArn': topic_arn,
            'Subject': subject[:100] if len(subject) > 100 else subject,
            'Message': message,
        }
        if message_attributes:
            kwargs['MessageAttributes'] = message_attributes

        response = clients.sns.publish(**kwargs)
        return response.get('MessageId', '')

    except ClientError as e:
        raise Exception(f"Failed to publish SNS notification: {e}")
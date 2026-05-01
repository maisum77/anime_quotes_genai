"""
S3 Storage Manager
------------------
Comprehensive S3 operations for the Anime Quote Generator pipeline:
1. Bucket structure management (inputs, outputs, models, logs, cache)
2. Upload/download with metadata and versioning
3. Presigned URL generation for direct access
4. Lifecycle management and cleanup
5. Batch operations for multiple objects
6. Content deduplication via hashing

S3 Bucket Structure:
    anime-quote-generator-{env}/
    ├── inputs/           # Raw input requests
    │   └── {yyyy}/{mm}/{dd}/{job_id}.json
    ├── outputs/          # Formatted generation results
    │   └── {yyyy}/{mm}/{dd}/{job_id}.json
    ├── models/           # GPT-2 model artifacts
    │   ├── model.bin
    │   ├── tokenizer/
    │   └── config.json
    ├── logs/             # Execution logs and audit trail
    │   └── {yyyy}/{mm}/{dd}/{job_id}.log
    ├── cache/            # Cached generation results
    │   └── {content_hash}.json
    ├── templates/        # Prompt templates and configurations
    │   └── {template_name}.json
    └── metrics/          # Aggregated metrics snapshots
        └── {yyyy}/{mm}/{dd}/metrics.json
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

from .aws_clients import clients
from .constants import (
    S3_BUCKET_PREFIX, S3_INPUTS_PREFIX, S3_OUTPUTS_PREFIX,
    S3_MODELS_PREFIX, S3_LOGS_PREFIX, S3_CACHE_PREFIX,
    S3_TEMPLATES_PREFIX, S3_METRICS_PREFIX
)

logger = logging.getLogger(__name__)

# S3 storage classes
STORAGE_CLASS_STANDARD = "STANDARD"
STORAGE_CLASS_INFREQUENT_ACCESS = "STANDARD_IA"
STORAGE_CLASS_GLACIER = "GLACIER"

# Content types
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_TEXT = "text/plain"
CONTENT_TYPE_BINARY = "application/octet-stream"


def get_s3_bucket_name(environment: Optional[str] = None) -> str:
    """
    Get S3 bucket name for the given environment.

    Args:
        environment: Environment name (dev, staging, prod). Defaults to
            ENVIRONMENT env var or 'dev'.

    Returns:
        S3 bucket name string
    """
    if not environment:
        environment = os.environ.get('ENVIRONMENT', 'dev')
    return f"{S3_BUCKET_PREFIX}-{environment}"


def generate_s3_key(
    prefix: str,
    job_id: str,
    filename: Optional[str] = None,
    extension: str = ".json",
    timestamp: Optional[int] = None
) -> str:
    """
    Generate a date-partitioned S3 key.

    Args:
        prefix: S3 prefix (inputs, outputs, models, logs, cache)
        job_id: Unique job identifier
        filename: Optional custom filename (defaults to job_id)
        extension: File extension (default .json)
        timestamp: Optional timestamp for date partitioning

    Returns:
        S3 key string in format: {prefix}/{yyyy}/{mm}/{dd}/{filename}{extension}
    """
    if timestamp is None:
        timestamp = int(time.time())

    date_str = time.strftime('%Y/%m/%d', time.gmtime(timestamp))
    name = filename or job_id

    return f"{prefix}/{date_str}/{name}{extension}"


def compute_content_hash(content: str) -> str:
    """
    Compute MD5 hash of content for deduplication.

    Args:
        content: String content to hash

    Returns:
        Hex digest of MD5 hash
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Upload Operations
# ---------------------------------------------------------------------------

def upload_to_s3(
    data: str,
    key: str,
    bucket: Optional[str] = None,
    content_type: str = CONTENT_TYPE_JSON,
    metadata: Optional[Dict[str, str]] = None,
    storage_class: str = STORAGE_CLASS_STANDARD,
    tagging: Optional[Dict[str, str]] = None
) -> str:
    """
    Upload string data to S3 with optional metadata and tagging.

    Args:
        data: String data to upload
        key: S3 object key
        bucket: S3 bucket name (uses default if None)
        content_type: MIME content type
        metadata: Optional custom metadata dict
        storage_class: S3 storage class (STANDARD, STANDARD_IA, GLACIER)
        tagging: Optional key-value tags

    Returns:
        S3 object URL

    Raises:
        Exception: If upload fails
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        extra_args: Dict[str, Any] = {
            'ContentType': content_type,
            'StorageClass': storage_class,
        }

        if metadata:
            # S3 metadata keys must be lowercase and match [a-z0-9-]
            sanitized = {k.lower().replace('_', '-'): str(v) for k, v in metadata.items()}
            extra_args['Metadata'] = sanitized

        if tagging:
            tag_set = '&'.join(f"{k}={v}" for k, v in tagging.items())
            extra_args['Tagging'] = tag_set

        clients.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data.encode('utf-8') if isinstance(data, str) else data,
            **extra_args
        )

        region = os.environ.get('AWS_REGION', 'us-east-1')
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

        logger.info("Uploaded s3://%s/%s (%d bytes)", bucket, key, len(data))
        return url

    except ClientError as e:
        logger.error("Failed to upload to S3: %s", e)
        raise Exception(f"Failed to upload to S3: {e}")


def upload_json(
    data: Dict[str, Any],
    key: str,
    bucket: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    indent: int = 2
) -> str:
    """
    Upload a Python dict as JSON to S3.

    Args:
        data: Dictionary to serialize and upload
        key: S3 object key
        bucket: S3 bucket name
        metadata: Optional custom metadata
        indent: JSON indentation level

    Returns:
        S3 object URL
    """
    return upload_to_s3(
        data=json.dumps(data, indent=indent, default=str),
        key=key,
        bucket=bucket,
        content_type=CONTENT_TYPE_JSON,
        metadata=metadata
    )


def upload_input_request(
    job_id: str,
    request_data: Dict[str, Any],
    bucket: Optional[str] = None
) -> str:
    """
    Store a preprocessed input request in S3.

    Args:
        job_id: Unique job identifier
        request_data: Validated request payload
        bucket: S3 bucket name

    Returns:
        S3 key where input is stored
    """
    key = generate_s3_key(S3_INPUTS_PREFIX, job_id, extension=".json")
    upload_json(
        data=request_data,
        key=key,
        bucket=bucket,
        metadata={"job-id": job_id, "content-type": "input-request"}
    )
    return key


def upload_output_result(
    job_id: str,
    result_data: Dict[str, Any],
    bucket: Optional[str] = None
) -> str:
    """
    Store a generation output result in S3.

    Args:
        job_id: Unique job identifier
        result_data: Formatted output result
        bucket: S3 bucket name

    Returns:
        S3 key where output is stored
    """
    key = generate_s3_key(S3_OUTPUTS_PREFIX, job_id, extension=".json")
    upload_json(
        data=result_data,
        key=key,
        bucket=bucket,
        metadata={"job-id": job_id, "content-type": "output-result"}
    )
    return key


def upload_execution_log(
    job_id: str,
    log_data: Dict[str, Any],
    bucket: Optional[str] = None
) -> str:
    """
    Store execution log data in S3.

    Args:
        job_id: Unique job identifier
        log_data: Log data dictionary
        bucket: S3 bucket name

    Returns:
        S3 key where log is stored
    """
    key = generate_s3_key(S3_LOGS_PREFIX, job_id, extension=".log")
    upload_json(
        data=log_data,
        key=key,
        bucket=bucket,
        metadata={"job-id": job_id, "content-type": "execution-log"},
        storage_class=STORAGE_CLASS_INFREQUENT_ACCESS
    )
    return key


def upload_cached_result(
    content: str,
    result_data: Dict[str, Any],
    bucket: Optional[str] = None,
    ttl_seconds: int = 3600
) -> str:
    """
    Cache a generation result keyed by content hash for deduplication.

    Args:
        content: Original generated text content
        result_data: Full result to cache
        bucket: S3 bucket name
        ttl_seconds: Cache time-to-live in seconds

    Returns:
        S3 key where cache entry is stored
    """
    content_hash = compute_content_hash(content)
    key = f"{S3_CACHE_PREFIX}/{content_hash}.json"

    cache_entry = {
        "content_hash": content_hash,
        "result": result_data,
        "cached_at": int(time.time()),
        "expires_at": int(time.time()) + ttl_seconds
    }

    upload_json(
        data=cache_entry,
        key=key,
        bucket=bucket,
        metadata={"content-hash": content_hash, "content-type": "cached-result"},
        tagging={"cache": "true", "ttl": str(ttl_seconds)}
    )
    return key


def upload_batch(
    items: List[Dict[str, Any]],
    prefix: str,
    bucket: Optional[str] = None
) -> List[str]:
    """
    Upload multiple items to S3, each keyed by its job_id.

    Args:
        items: List of dicts, each must contain 'job_id' and 'data' keys
        prefix: S3 prefix for all items
        bucket: S3 bucket name

    Returns:
        List of S3 keys uploaded
    """
    keys = []
    for item in items:
        job_id = item.get("job_id", f"unknown-{int(time.time())}")
        data = item.get("data", {})
        key = generate_s3_key(prefix, job_id, extension=".json")
        upload_json(data=data, key=key, bucket=bucket, metadata={"job-id": job_id})
        keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Download Operations
# ---------------------------------------------------------------------------

def download_from_s3(
    key: str,
    bucket: Optional[str] = None
) -> str:
    """
    Download string data from S3.

    Args:
        key: S3 object key
        bucket: S3 bucket name

    Returns:
        Downloaded data as string

    Raises:
        Exception: If download fails
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        response = clients.s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')
        logger.info("Downloaded s3://%s/%s (%d bytes)", bucket, key, len(data))
        return data
    except ClientError as e:
        logger.error("Failed to download from S3: %s", e)
        raise Exception(f"Failed to download from S3: {e}")


def download_json(
    key: str,
    bucket: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download and parse a JSON object from S3.

    Args:
        key: S3 object key
        bucket: S3 bucket name

    Returns:
        Parsed JSON as dictionary
    """
    raw = download_from_s3(key, bucket)
    return json.loads(raw)


def download_input_request(
    job_id: str,
    bucket: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Download an input request by job ID.

    Args:
        job_id: Job identifier
        bucket: S3 bucket name

    Returns:
        Input request data or None if not found
    """
    key = generate_s3_key(S3_INPUTS_PREFIX, job_id, extension=".json")
    try:
        return download_json(key, bucket)
    except Exception:
        return None


def download_output_result(
    job_id: str,
    bucket: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Download an output result by job ID.

    Args:
        job_id: Job identifier
        bucket: S3 bucket name

    Returns:
        Output result data or None if not found
    """
    key = generate_s3_key(S3_OUTPUTS_PREFIX, job_id, extension=".json")
    try:
        return download_json(key, bucket)
    except Exception:
        return None


def check_cached_result(
    content: str,
    bucket: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Check if a cached result exists for the given content.

    Args:
        content: Generated text content to look up
        bucket: S3 bucket name

    Returns:
        Cached result data or None if not found / expired
    """
    content_hash = compute_content_hash(content)
    key = f"{S3_CACHE_PREFIX}/{content_hash}.json"

    try:
        cached = download_json(key, bucket)
        expires_at = cached.get("expires_at", 0)

        if expires_at > int(time.time()):
            logger.info("Cache hit for hash %s", content_hash[:8])
            return cached.get("result")

        # Cache expired - log and return None
        logger.info("Cache expired for hash %s", content_hash[:8])
        return None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Presigned URL Operations
# ---------------------------------------------------------------------------

def generate_presigned_download_url(
    key: str,
    bucket: Optional[str] = None,
    expiration: int = 3600
) -> str:
    """
    Generate a presigned URL for downloading an S3 object.

    Args:
        key: S3 object key
        bucket: S3 bucket name
        expiration: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL string

    Raises:
        Exception: If URL generation fails
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        url = clients.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        logger.info("Generated presigned download URL for %s (expires in %ds)", key, expiration)
        return url
    except ClientError as e:
        logger.error("Failed to generate presigned URL: %s", e)
        raise Exception(f"Failed to generate presigned URL: {e}")


def generate_presigned_upload_url(
    key: str,
    bucket: Optional[str] = None,
    expiration: int = 3600,
    content_type: str = CONTENT_TYPE_JSON
) -> str:
    """
    Generate a presigned URL for uploading to S3.

    Args:
        key: S3 object key
        bucket: S3 bucket name
        expiration: URL expiration time in seconds
        content_type: Expected content type

    Returns:
        Presigned URL string

    Raises:
        Exception: If URL generation fails
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        url = clients.s3.generate_presigned_url(
            'put_object',
            Params={'Bucket': bucket, 'Key': key, 'ContentType': content_type},
            ExpiresIn=expiration
        )
        logger.info("Generated presigned upload URL for %s (expires in %ds)", key, expiration)
        return url
    except ClientError as e:
        logger.error("Failed to generate presigned upload URL: %s", e)
        raise Exception(f"Failed to generate presigned upload URL: {e}")


# ---------------------------------------------------------------------------
# List and Search Operations
# ---------------------------------------------------------------------------

def list_s3_objects(
    prefix: str,
    bucket: Optional[str] = None,
    max_keys: int = 100,
    continuation_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    List objects in S3 bucket with given prefix.

    Args:
        prefix: S3 prefix to filter
        bucket: S3 bucket name
        max_keys: Maximum number of keys to return
        continuation_token: Token for paginated results

    Returns:
        Dict with 'objects' list and 'is_truncated' flag
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        kwargs: Dict[str, Any] = {
            'Bucket': bucket,
            'Prefix': prefix,
            'MaxKeys': max_keys
        }
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token

        response = clients.s3.list_objects_v2(**kwargs)

        objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"'),
                    'storage_class': obj.get('StorageClass', STORAGE_CLASS_STANDARD)
                })

        return {
            'objects': objects,
            'is_truncated': response.get('IsTruncated', False),
            'next_continuation_token': response.get('NextContinuationToken'),
            'key_count': response.get('KeyCount', 0)
        }

    except ClientError as e:
        logger.error("Failed to list S3 objects: %s", e)
        raise Exception(f"Failed to list S3 objects: {e}")


def list_outputs_by_date(
    date: Optional[str] = None,
    bucket: Optional[str] = None,
    max_keys: int = 100
) -> List[Dict[str, Any]]:
    """
    List output results for a specific date.

    Args:
        date: Date string in YYYY/MM/DD format (defaults to today)
        bucket: S3 bucket name
        max_keys: Maximum results

    Returns:
        List of output object metadata
    """
    if date is None:
        date = datetime.utcnow().strftime('%Y/%m/%d')

    prefix = f"{S3_OUTPUTS_PREFIX}/{date}"
    result = list_s3_objects(prefix, bucket, max_keys)
    return result.get('objects', [])


def get_object_metadata(
    key: str,
    bucket: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get metadata for an S3 object without downloading it.

    Args:
        key: S3 object key
        bucket: S3 bucket name

    Returns:
        Object metadata dict or None if not found
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        response = clients.s3.head_object(Bucket=bucket, Key=key)
        return {
            'content_type': response.get('ContentType', ''),
            'content_length': response.get('ContentLength', 0),
            'last_modified': response.get('LastModified', '').isoformat() if response.get('LastModified') else '',
            'etag': response.get('ETag', '').strip('"'),
            'metadata': response.get('Metadata', {}),
            'storage_class': response.get('StorageClass', STORAGE_CLASS_STANDARD)
        }
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None
        logger.error("Failed to get object metadata: %s", e)
        raise Exception(f"Failed to get object metadata: {e}")


def object_exists(
    key: str,
    bucket: Optional[str] = None
) -> bool:
    """
    Check if an S3 object exists.

    Args:
        key: S3 object key
        bucket: S3 bucket name

    Returns:
        True if object exists, False otherwise
    """
    return get_object_metadata(key, bucket) is not None


# ---------------------------------------------------------------------------
# Delete and Lifecycle Operations
# ---------------------------------------------------------------------------

def delete_s3_object(
    key: str,
    bucket: Optional[str] = None
) -> None:
    """
    Delete a single object from S3.

    Args:
        key: S3 object key
        bucket: S3 bucket name
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        clients.s3.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted s3://%s/%s", bucket, key)
    except ClientError as e:
        logger.error("Failed to delete S3 object: %s", e)
        raise Exception(f"Failed to delete S3 object: {e}")


def delete_s3_objects(
    keys: List[str],
    bucket: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete multiple objects from S3 in a batch.

    Args:
        keys: List of S3 object keys
        bucket: S3 bucket name

    Returns:
        Dict with 'deleted' count and 'errors' list
    """
    if bucket is None:
        bucket = get_s3_bucket_name()

    if not keys:
        return {'deleted': 0, 'errors': []}

    try:
        # S3 batch delete supports up to 1000 keys per request
        all_deleted = 0
        all_errors = []

        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            response = clients.s3.delete_objects(
                Bucket=bucket,
                Delete={
                    'Objects': [{'Key': k} for k in batch],
                    'Quiet': False
                }
            )

            all_deleted += len(response.get('Deleted', []))
            all_errors.extend([
                {'key': err.get('Key'), 'code': err.get('Code'), 'message': err.get('Message')}
                for err in response.get('Errors', [])
            ])

        logger.info("Batch deleted %d objects, %d errors", all_deleted, len(all_errors))
        return {'deleted': all_deleted, 'errors': all_errors}

    except ClientError as e:
        logger.error("Failed to batch delete S3 objects: %s", e)
        raise Exception(f"Failed to batch delete S3 objects: {e}")


def cleanup_expired_cache(
    bucket: Optional[str] = None,
    max_keys: int = 1000
) -> int:
    """
    Remove expired cache entries from S3.

    Args:
        bucket: S3 bucket name
        max_keys: Maximum number of cache entries to scan

    Returns:
        Number of expired entries deleted
    """
    now = int(time.time())
    deleted_count = 0

    result = list_s3_objects(S3_CACHE_PREFIX, bucket, max_keys)

    for obj in result.get('objects', []):
        try:
            cached = download_json(obj['key'], bucket)
            expires_at = cached.get("expires_at", 0)

            if expires_at <= now:
                delete_s3_object(obj['key'], bucket)
                deleted_count += 1
        except Exception as e:
            logger.warning("Failed to process cache entry %s: %s", obj['key'], e)

    logger.info("Cleaned up %d expired cache entries", deleted_count)
    return deleted_count


def copy_s3_object(
    source_key: str,
    dest_key: str,
    source_bucket: Optional[str] = None,
    dest_bucket: Optional[str] = None,
    storage_class: str = STORAGE_CLASS_STANDARD
) -> str:
    """
    Copy an S3 object to a new location.

    Args:
        source_key: Source S3 object key
        dest_key: Destination S3 object key
        source_bucket: Source bucket (defaults to project bucket)
        dest_bucket: Destination bucket (defaults to project bucket)
        storage_class: Storage class for the copy

    Returns:
        Destination S3 URL
    """
    if source_bucket is None:
        source_bucket = get_s3_bucket_name()
    if dest_bucket is None:
        dest_bucket = source_bucket

    try:
        copy_source = {'Bucket': source_bucket, 'Key': source_key}

        clients.s3.copy_object(
            CopySource=copy_source,
            Bucket=dest_bucket,
            Key=dest_key,
            StorageClass=storage_class
        )

        region = os.environ.get('AWS_REGION', 'us-east-1')
        url = f"https://{dest_bucket}.s3.{region}.amazonaws.com/{dest_key}"
        logger.info("Copied s3://%s/%s -> s3://%s/%s", source_bucket, source_key, dest_bucket, dest_key)
        return url

    except ClientError as e:
        logger.error("Failed to copy S3 object: %s", e)
        raise Exception(f"Failed to copy S3 object: {e}")


# ---------------------------------------------------------------------------
# Model Storage Operations
# ---------------------------------------------------------------------------

def upload_model_artifact(
    artifact_name: str,
    data: bytes,
    content_type: str = CONTENT_TYPE_BINARY,
    bucket: Optional[str] = None
) -> str:
    """
    Upload a GPT-2 model artifact to the models prefix.

    Args:
        artifact_name: Name of the artifact (e.g., 'model.bin', 'config.json')
        data: Binary artifact data
        content_type: MIME type
        bucket: S3 bucket name

    Returns:
        S3 key where artifact is stored
    """
    key = f"{S3_MODELS_PREFIX}/{artifact_name}"

    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        clients.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type
        )
        logger.info("Uploaded model artifact: %s", key)
        return key
    except ClientError as e:
        logger.error("Failed to upload model artifact: %s", e)
        raise Exception(f"Failed to upload model artifact: {e}")


def download_model_artifact(
    artifact_name: str,
    bucket: Optional[str] = None
) -> bytes:
    """
    Download a GPT-2 model artifact from S3.

    Args:
        artifact_name: Name of the artifact
        bucket: S3 bucket name

    Returns:
        Binary artifact data
    """
    key = f"{S3_MODELS_PREFIX}/{artifact_name}"

    if bucket is None:
        bucket = get_s3_bucket_name()

    try:
        response = clients.s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read()
        logger.info("Downloaded model artifact: %s (%d bytes)", key, len(data))
        return data
    except ClientError as e:
        logger.error("Failed to download model artifact: %s", e)
        raise Exception(f"Failed to download model artifact: {e}")


def list_model_artifacts(
    bucket: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all model artifacts in S3.

    Args:
        bucket: S3 bucket name

    Returns:
        List of model artifact metadata
    """
    result = list_s3_objects(S3_MODELS_PREFIX, bucket, max_keys=100)
    return result.get('objects', [])


def get_s3_client():
    """Get the underlying S3 client for advanced operations."""
    return clients.s3

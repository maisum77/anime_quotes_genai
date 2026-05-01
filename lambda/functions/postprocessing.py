"""
Postprocessing Lambda Function
-----------------------------
Processes generated content from the generation Lambda:
1. Formats and structures the output
2. Stores results in S3
3. Updates DynamoDB with final status
4. Sends notifications via SNS
5. Updates user history

Environment Variables:
- S3_BUCKET: S3 bucket for storing outputs
- JOBS_TABLE: DynamoDB table for job tracking
- HISTORY_TABLE: DynamoDB table for user history
- SNS_GENERATION_COMPLETE: SNS topic for completion notifications
- SNS_GENERATION_FAILED: SNS topic for failure notifications
"""

import json
import os
import time
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

# Import shared modules
from shared import constants
from shared.logging import logger, log_metric
from shared.dynamodb_manager import update_job_status, store_user_history
from shared.s3_storage import upload_to_s3
from shared.sns_manager import publish_notification


def format_content(generation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format and structure the generated content for final output.
    
    Args:
        generation_result: Raw generation result from generation Lambda
        
    Returns:
        Formatted output with metadata
    """
    speech_type = generation_result.get("speech_type", constants.DEFAULT_SPEECH_TYPE)
    generation_type = generation_result.get("generation_type", constants.DEFAULT_GENERATION_TYPE)
    content = generation_result.get("content", "")
    generation_method = generation_result.get("generation_method", "unknown")
    characters = generation_result.get("characters", constants.DEFAULT_CHARACTERS)
    temperature = generation_result.get("temperature", constants.DEFAULT_TEMPERATURE)
    
    # Calculate content metrics
    word_count = len(content.split())
    char_count = len(content)
    line_count = content.count('\n') + 1
    
    # Generate content hash for deduplication
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    
    # Format based on generation type
    if generation_type == "dialogue":
        # Parse dialogue lines
        lines = content.strip().split('\n')
        formatted_dialogue = []
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                char, dialogue = line.split(':', 1)
                formatted_dialogue.append({
                    "character": char.strip(),
                    "dialogue": dialogue.strip()
                })
            else:
                formatted_dialogue.append({
                    "character": "Unknown",
                    "dialogue": line
                })
        
        formatted_content = {
            "type": "dialogue",
            "characters": characters,
            "exchanges": len(formatted_dialogue),
            "dialogue": formatted_dialogue,
            "raw_text": content
        }
    else:
        # Speech formatting
        formatted_content = {
            "type": "speech",
            "speech_type": speech_type,
            "text": content,
            "paragraphs": [p.strip() for p in content.split('\n\n') if p.strip()]
        }
    
    return {
        "formatted_content": formatted_content,
        "metadata": {
            "speech_type": speech_type,
            "generation_type": generation_type,
            "generation_method": generation_method,
            "temperature": temperature,
            "word_count": word_count,
            "character_count": char_count,
            "line_count": line_count,
            "content_hash": content_hash,
            "timestamp": int(time.time()),
            "formatted_at": datetime.utcnow().isoformat() + "Z"
        }
    }


def store_output_in_s3(job_id: str, formatted_output: Dict[str, Any], request_data: Dict[str, Any]) -> str:
    """
    Store formatted output in S3.
    
    Args:
        job_id: Unique job identifier
        formatted_output: Formatted content and metadata
        request_data: Original request data
        
    Returns:
        S3 key where output is stored
    """
    bucket = os.environ.get("S3_BUCKET", "")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is not set.")
    
    # Generate S3 key
    timestamp = datetime.utcnow().strftime("%Y/%m/%d")
    s3_key = f"{constants.S3_OUTPUTS_PREFIX}/{timestamp}/{job_id}.json"
    
    # Create complete output document
    output_document = {
        "job_id": job_id,
        "status": "completed",
        "request": request_data,
        "output": formatted_output,
        "storage": {
            "bucket": bucket,
            "key": s3_key,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
    
    # Upload to S3 using s3_storage.upload_to_s3 (data=, key=, bucket=)
    upload_to_s3(
        data=json.dumps(output_document, indent=2),
        key=s3_key,
        bucket=bucket,
        content_type="application/json"
    )
    
    logger.info(f"Stored output for job {job_id} in S3: s3://{bucket}/{s3_key}")
    
    return s3_key


def update_final_job_status(job_id: str, s3_key: str, formatted_output: Dict[str, Any]) -> None:
    """
    Update job status in DynamoDB with final completion details.
    
    Args:
        job_id: Unique job identifier
        s3_key: S3 key where output is stored
        formatted_output: Formatted content and metadata
    """
    metadata = formatted_output.get("metadata", {})
    
    update_job_status(
        job_id=job_id,
        status="completed",
        metadata={
            "completed_at": int(time.time()),
            "s3_key": s3_key,
            "word_count": metadata.get("word_count", 0),
            "generation_method": metadata.get("generation_method", "unknown"),
            "content_hash": metadata.get("content_hash", ""),
            "final_status": "success"
        }
    )
    
    logger.info(f"Updated final job status for {job_id}")


def store_in_user_history(job_id: str, user_id: str, request_data: Dict[str, Any], 
                         formatted_output: Dict[str, Any], s3_key: str) -> None:
    """
    Store generation in user history table.
    
    Args:
        job_id: Unique job identifier
        user_id: User identifier (from request or API key)
        request_data: Original request data
        formatted_output: Formatted content and metadata
        s3_key: S3 key where output is stored
    """
    if not user_id:
        logger.warning("No user_id provided, skipping history storage")
        return
    
    metadata = formatted_output.get("metadata", {})
    
    history_item = {
        "user_id": user_id,
        "job_id": job_id,
        "timestamp": int(time.time()),
        "request": request_data,
        "result": {
            "speech_type": metadata.get("speech_type"),
            "generation_type": metadata.get("generation_type"),
            "generation_method": metadata.get("generation_method"),
            "word_count": metadata.get("word_count", 0),
            "content_preview": formatted_output.get("formatted_content", {}).get("text", "")[:100] + "...",
            "s3_key": s3_key
        }
    }
    
    store_user_history(history_item)
    logger.info(f"Stored generation in history for user {user_id}, job {job_id}")


def send_notifications(job_id: str, user_id: str, request_data: Dict[str, Any],
                      formatted_output: Dict[str, Any], s3_key: str) -> None:
    """
    Send notifications via SNS.
    
    Args:
        job_id: Unique job identifier
        user_id: User identifier
        request_data: Original request data
        formatted_output: Formatted content and metadata
        s3_key: S3 key where output is stored
    """
    metadata = formatted_output.get("metadata", {})
    
    # Success notification via sns_manager.publish_notification
    notification = {
        "event": "generation_completed",
        "job_id": job_id,
        "user_id": user_id or "anonymous",
        "timestamp": int(time.time()),
        "result": {
            "speech_type": metadata.get("speech_type"),
            "generation_type": metadata.get("generation_type"),
            "generation_method": metadata.get("generation_method"),
            "word_count": metadata.get("word_count", 0),
            "s3_key": s3_key
        },
        "request_summary": {
            "speech_type": request_data.get("speech_type"),
            "generation_type": request_data.get("generation_type"),
            "has_custom_prompt": bool(request_data.get("custom_prompt"))
        }
    }
    
    try:
        publish_notification(
            topic_name=constants.SNS_GENERATION_COMPLETE,
            subject=f"Generation Completed: {job_id}",
            message=notification,
            event_type=constants.EVENT_GENERATION_COMPLETED,
            severity=constants.SEVERITY_INFO,
            job_id=job_id,
            source="postprocessing"
        )
        logger.info(f"Sent success notification for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to send success notification for job {job_id}: {e}")


def process_generation_result(message_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single generation result from SQS.
    
    Args:
        message_body: Parsed JSON from SQS message
        
    Returns:
        Dict with processing results
    """
    job_id = message_body.get("job_id")
    generation_result = message_body.get("generation_result", {})
    request_data = message_body.get("request_data", {})
    
    if not job_id:
        raise ValueError("Missing job_id in message")
    
    logger.info(f"Processing postprocessing for job {job_id}")
    
    start_time = time.time()
    
    try:
        # Step 1: Format content
        formatted_output = format_content(generation_result)
        
        # Step 2: Store in S3
        s3_key = store_output_in_s3(job_id, formatted_output, request_data)
        
        # Step 3: Update final job status
        update_final_job_status(job_id, s3_key, formatted_output)
        
        # Step 4: Store in user history (if user_id available)
        user_id = request_data.get("user_id", "") or request_data.get("api_key", "")
        store_in_user_history(job_id, user_id, request_data, formatted_output, s3_key)
        
        # Step 5: Send notifications
        send_notifications(job_id, user_id, request_data, formatted_output, s3_key)
        
        processing_time = time.time() - start_time
        
        # Log metrics
        log_metric("PostprocessingTime", processing_time, unit="Seconds")
        log_metric("OutputWordCount", formatted_output.get("metadata", {}).get("word_count", 0), unit="Count")
        
        logger.info(f"Successfully processed job {job_id} in {processing_time:.2f}s")
        
        return {
            "job_id": job_id,
            "status": "success",
            "s3_key": s3_key,
            "processing_time": processing_time,
            "word_count": formatted_output.get("metadata", {}).get("word_count", 0),
            "generation_method": generation_result.get("generation_method", "unknown")
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Postprocessing failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        
        # Update job status to "postprocessing_failed"
        update_job_status(
            job_id=job_id,
            status="postprocessing_failed",
            metadata={
                "failed_at": int(time.time()),
                "processing_time": processing_time,
                "error": str(e)
            }
        )
        
        # Send failure notification via sns_manager.publish_notification
        try:
            publish_notification(
                topic_name=constants.SNS_GENERATION_FAILED,
                subject=f"Postprocessing Failed: {job_id}",
                message={
                    "event": "postprocessing_failed",
                    "job_id": job_id,
                    "timestamp": int(time.time()),
                    "error": str(e),
                    "processing_time": processing_time
                },
                event_type=constants.EVENT_POSTPROCESSING_FAILED,
                severity=constants.SEVERITY_ERROR,
                job_id=job_id,
                source="postprocessing"
            )
        except Exception as notif_err:
            logger.warning(f"Failed to send failure notification for job {job_id}: {notif_err}")
        
        # Log error metric
        log_metric("PostprocessingErrors", 1, unit="Count")
        
        raise


def lambda_handler(event, context):
    """
    AWS Lambda handler for processing SQS messages from generation queue.
    
    Expected event format (SQS trigger):
    {
        "Records": [
            {
                "messageId": "...",
                "body": "{\"job_id\": \"...\", \"generation_result\": {...}, \"request_data\": {...}}",
                ...
            }
        ]
    }
    """
    logger.info(f"Received event with {len(event.get('Records', []))} records")
    
    successful_messages = []
    failed_messages = []
    
    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        
        try:
            # Parse message body
            message_body = json.loads(record.get("body", "{}"))
            
            # Process the message
            result = process_generation_result(message_body)
            
            successful_messages.append({
                "message_id": message_id,
                "job_id": result.get("job_id"),
                "s3_key": result.get("s3_key")
            })
            
        except Exception as e:
            error_msg = f"Failed to process message {message_id}: {str(e)}"
            logger.error(error_msg)
            failed_messages.append({"message_id": message_id, "error": str(e)})
    
    # Log summary
    logger.info(f"Processed {len(successful_messages)} messages successfully, {len(failed_messages)} failed")
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "successful": len(successful_messages),
            "failed": len(failed_messages),
            "successful_details": successful_messages,
            "failed_details": failed_messages
        })
    }
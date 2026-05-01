"""
Generation Lambda Function
-------------------------
Processes SQS messages from the generation queue and generates anime speeches/dialogues
using a three-tier strategy: Gemini → GPT-2 → Fallback quotes.

Environment Variables:
- GOOGLE_API_KEY: API key for Gemini (optional)
- S3_BUCKET: S3 bucket for storing models and outputs
- JOBS_TABLE: DynamoDB table for job tracking
- POSTPROCESSING_QUEUE: SQS queue for forwarding results to postprocessing
- GPT2_MODEL_PATH: S3 path to GPT-2 model (optional)
"""

import json
import os
import random
import time
from typing import Dict, Any, Optional

# Import shared modules
from shared import constants
from shared.logging import logger, log_metric
from shared.validation import validate_speech_type, validate_generation_type
from shared.dynamodb_manager import update_job_status
from shared.sqs_manager import send_postprocessing_request

# Fallback prompts (same as in original handler)
FALLBACK_PROMPTS = {
    "motivational": [
        "Listen everyone! True strength comes from within — never give up on your dreams!",
        "Stand up! Even if you fall a thousand times, rise a thousand and one!",
        "The path to victory is paved with the tears of those who never quit!",
    ],
    "battle": [
        "This battle isn't over! My power has no limits when I fight for my friends!",
        "You think you've won? I'm just getting started — prepare yourself!",
        "My true strength awakens when everything is on the line!",
    ],
    "friendship": [
        "We're not alone — together our bonds make us unstoppable!",
        "My friends are my greatest power; nothing can break what we've built!",
        "Side by side, we face every challenge that comes our way!",
    ],
    "determination": [
        "I made a promise I intend to keep, no matter the cost!",
        "Nothing will stop me from reaching my goal — not pain, not fear!",
        "Even if I fall, I'll keep moving forward until my last breath!",
    ],
    "villain": [
        "You fools cling to hope while the world crumbles beneath your feet!",
        "Power is the only truth — and I have claimed it all!",
        "Your resistance only delays the inevitable. Bow before me!",
    ],
}


def _build_description(generation_type: str, speech_type: str, custom_prompt: str, characters: list) -> str:
    """Return a CrewAI task description string."""
    if generation_type == "dialogue":
        char_list = ", ".join(characters)
        return (
            f"Write a dramatic anime-style dialogue between {char_list}. "
            "Each character should have a distinct voice. Show 3 exchanges total. "
            "Make it feel like a pivotal scene from an epic anime series. "
            "Format: 'CHARACTER: dialogue line' on separate lines."
        )

    if custom_prompt:
        return (
            f"Write an anime-style speech based on this prompt: {custom_prompt}. "
            "Make it dramatic, emotional, and around 150-250 words."
        )

    descriptions = {
        "motivational": (
            "Write a powerful motivational speech (150-250 words) from an anime protagonist "
            "inspiring their team before a critical, world-changing battle."
        ),
        "battle": (
            "Write an intense battle speech (150-250 words) from an anime warrior "
            "facing their strongest opponent with everything on the line."
        ),
        "friendship": (
            "Write a heartfelt speech (150-250 words) about the power of friendship "
            "from an anime character addressing their most trusted companions."
        ),
        "determination": (
            "Write a speech about unwavering determination (150-250 words) from an anime hero "
            "who refuses to give up against impossible odds."
        ),
        "villain": (
            "Write a menacing villain monologue (150-250 words) from an anime antagonist "
            "explaining their twisted vision for remaking the world."
        ),
    }
    return descriptions.get(speech_type, descriptions["motivational"])


def _generate_with_gemini(generation_type: str, speech_type: str, custom_prompt: str, characters: list) -> str:
    """Call Gemini via CrewAI and return the generated text."""
    try:
        # Import here so Lambda cold-start is faster when running in fallback mode
        from crewai import Agent, Task, Crew, LLM  # noqa: PLC0415
    except ImportError:
        raise ImportError("crewai not installed")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    gemini_llm = LLM(
        model="gemini/gemini-1.5-flash",
        api_key=api_key,
    )

    agent = Agent(
        role="Anime Speech Writer",
        goal="Write compelling, dramatic anime-style speeches and dialogues",
        backstory=(
            "You are a legendary anime scriptwriter who crafted the most iconic speeches "
            "in anime history. Your words make audiences laugh, cry, and cheer."
        ),
        llm=gemini_llm,
        verbose=False,
    )

    task = Task(
        description=_build_description(generation_type, speech_type, custom_prompt, characters),
        agent=agent,
        expected_output="A dramatic anime-style speech or dialogue (150-250 words)",
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)


def _generate_with_gpt2(speech_type: str, temperature: float = 0.8, max_length: int = 200) -> str:
    """Generate anime speech using GPT-2 fine-tuned model."""
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        raise ImportError("transformers or torch not installed")

    # Check if model is available in S3
    model_path = os.environ.get("GPT2_MODEL_PATH", "")
    if not model_path:
        raise ValueError("GPT2_MODEL_PATH environment variable is not set.")

    # Download model from S3 if not already cached
    # For now, we'll use a simplified approach - in production, you'd download and cache
    # the model in /tmp directory
    local_model_path = f"/tmp/gpt2_model"
    
    # Speech prompts based on type (from gen.py)
    prompts = {
        'motivational': [
            "Listen everyone! True strength",
            "Never give up! The path to victory",
            "I'll show you what it means to",
            "This is why we fight! Because",
            "Stand up and keep moving forward!"
        ],
        'battle': [
            "This battle isn't over!",
            "I won't let you win because",
            "My power comes from",
            "You think you've won? But",
            "This is my true strength!"
        ],
        'friendship': [
            "We're not alone because",
            "My friends are my power and",
            "Together we can overcome",
            "The bonds we share will",
            "You're all precious to me because"
        ],
        'determination': [
            "I made a promise to",
            "No matter what happens, I will",
            "Even if I fall, I'll keep",
            "My dream is to become",
            "Nothing will stop me from"
        ],
        'villain': [
            "You fools don't understand that",
            "Power is everything and",
            "The weak deserve to",
            "This world will bow before",
            "Your hope is meaningless because"
        ]
    }
    
    # Select prompt
    prompt = random.choice(prompts.get(speech_type, prompts['motivational']))
    
    # Load model (simplified - in production you'd implement proper caching)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load GPT-2 model: {e}")
    
    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate with parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add dramatic ending if not present
    endings = [
        " That's what makes us strong!",
        " And that's why we'll never lose!",
        " This is our path to victory!",
        " Believe it!",
        " That's my ninja way!",
        " Together, we're unstoppable!"
    ]
    
    if not generated_text.endswith(('!', '.', '?')):
        generated_text += random.choice(endings)
    
    return generated_text


def _generate_fallback(speech_type: str) -> str:
    """Return a pre-written fallback quote when all other methods fail."""
    pool = FALLBACK_PROMPTS.get(speech_type, FALLBACK_PROMPTS["motivational"])
    return random.choice(pool)


def generate_content(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate content using three-tier strategy:
    1. Try Gemini with CrewAI
    2. Fall back to GPT-2 if Gemini fails
    3. Use pre-written fallback quotes if both fail
    
    Returns:
        Dict with generated content and metadata
    """
    speech_type = request_data.get("speech_type", constants.DEFAULT_SPEECH_TYPE)
    generation_type = request_data.get("generation_type", constants.DEFAULT_GENERATION_TYPE)
    custom_prompt = request_data.get("custom_prompt", "")
    characters = request_data.get("characters", constants.DEFAULT_CHARACTERS)
    temperature = float(request_data.get("temperature", constants.DEFAULT_TEMPERATURE))
    
    logger.info(
        "Generating content | type=%s | speech_type=%s | custom_prompt=%s",
        generation_type, speech_type, bool(custom_prompt)
    )
    
    generation_method = "unknown"
    content = ""
    error_messages = []
    
    # Tier 1: Try Gemini
    try:
        content = _generate_with_gemini(generation_type, speech_type, custom_prompt, characters)
        generation_method = "gemini"
        logger.info("Successfully generated content with Gemini")
    except Exception as e:
        error_msg = f"Gemini generation failed: {str(e)}"
        error_messages.append(error_msg)
        logger.warning(error_msg)
        
        # Tier 2: Try GPT-2
        try:
            if generation_type == "speech":
                content = _generate_with_gpt2(speech_type, temperature)
                generation_method = "gpt2"
                logger.info("Successfully generated content with GPT-2")
            else:
                # GPT-2 doesn't support dialogue generation in this implementation
                raise ValueError("GPT-2 dialogue generation not implemented")
        except Exception as e2:
            error_msg = f"GPT-2 generation failed: {str(e2)}"
            error_messages.append(error_msg)
            logger.warning(error_msg)
            
            # Tier 3: Use fallback quotes
            if generation_type == "speech":
                content = _generate_fallback(speech_type)
                generation_method = "fallback"
                logger.info("Using fallback quotes")
            else:
                # For dialogue, create a simple fallback
                char_list = ", ".join(characters)
                content = f"{characters[0]}: I won't give up!\n{characters[1]}: You're still weak!\n{characters[0] if len(characters) > 0 else 'Hero'}: That's why I'll keep fighting!"
                generation_method = "fallback_dialogue"
    
    return {
        "speech_type": speech_type,
        "generation_type": generation_type,
        "content": content,
        "generation_method": generation_method,
        "characters": characters,
        "temperature": temperature,
        "errors": error_messages if error_messages else None,
        "timestamp": int(time.time())
    }


def process_sqs_message(message_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single SQS message containing generation request.
    
    Args:
        message_body: Parsed JSON from SQS message
        
    Returns:
        Dict with generation results
    """
    job_id = message_body.get("job_id")
    request_data = message_body.get("request_data", {})
    
    if not job_id:
        raise ValueError("Missing job_id in SQS message")
    
    logger.info(f"Processing generation job {job_id}")
    
    # Update job status to "generating"
    update_job_status(
        job_id=job_id,
        status="generating",
        metadata={"started_at": int(time.time())}
    )
    
    # Generate content
    start_time = time.time()
    try:
        result = generate_content(request_data)
        generation_time = time.time() - start_time
        
        # Update job status to "completed"
        update_job_status(
            job_id=job_id,
            status="completed",
            metadata={
                "completed_at": int(time.time()),
                "generation_time": generation_time,
                "generation_method": result.get("generation_method"),
                "content_length": len(result.get("content", ""))
            }
        )
        
        # Log metrics
        log_metric("GenerationTime", generation_time, unit="Seconds")
        log_metric("GenerationMethod", 1, unit="Count", dimensions={
            "method": result.get("generation_method", "unknown")
        })
        
        logger.info(f"Successfully generated content for job {job_id} in {generation_time:.2f}s")
        
        # Prepare result for postprocessing
        postprocessing_payload = {
            "job_id": job_id,
            "generation_result": result,
            "request_data": request_data,
            "timestamp": int(time.time())
        }
        
        return postprocessing_payload
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"Generation failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        
        # Update job status to "failed"
        update_job_status(
            job_id=job_id,
            status="failed",
            metadata={
                "failed_at": int(time.time()),
                "generation_time": generation_time,
                "error": str(e)
            }
        )
        
        # Log error metric
        log_metric("GenerationErrors", 1, unit="Count")
        
        raise


def lambda_handler(event, context):
    """
    AWS Lambda handler for processing SQS messages.
    
    Expected event format (SQS trigger):
    {
        "Records": [
            {
                "messageId": "...",
                "body": "{\"job_id\": \"...\", \"request_data\": {...}}",
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
            result = process_sqs_message(message_body)
            
            # Send to postprocessing queue via sqs_manager
            try:
                send_postprocessing_request(
                    job_id=result.get("job_id"),
                    generation_result=result.get("generation_result", {}),
                    request_data=result.get("request_data", {})
                )
                logger.info(f"Forwarded job {result.get('job_id')} to postprocessing queue")
            except Exception as sqs_err:
                logger.warning(f"Failed to forward job to postprocessing queue: {sqs_err}")
            
            successful_messages.append(message_id)
            
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
            "failed_details": failed_messages
        })
    }
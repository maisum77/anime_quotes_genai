"""
Input validation and sanitization utilities
"""
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from .constants import (
    SPEECH_TYPES, GENERATION_TYPES, MAX_BATCH_SIZE,
    ERROR_INVALID_SPEECH_TYPE, ERROR_INVALID_GENERATION_TYPE,
    ERROR_MISSING_REQUIRED_FIELD, ERROR_BATCH_SIZE_EXCEEDED
)

def validate_speech_type(speech_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate speech_type parameter
    
    Args:
        speech_type: Speech type to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if speech_type not in SPEECH_TYPES:
        return False, ERROR_INVALID_SPEECH_TYPE
    return True, None

def validate_generation_type(generation_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate generation_type parameter
    
    Args:
        generation_type: Generation type to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if generation_type not in GENERATION_TYPES:
        return False, ERROR_INVALID_GENERATION_TYPE
    return True, None

def validate_characters(characters: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate characters list
    
    Args:
        characters: List of character names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(characters, list):
        return False, "characters must be a list"
    
    if len(characters) < 2:
        return False, "At least 2 characters required for dialogue"
    
    for char in characters:
        if not isinstance(char, str):
            return False, "All character names must be strings"
        if len(char.strip()) == 0:
            return False, "Character names cannot be empty"
        if len(char) > 100:
            return False, "Character names cannot exceed 100 characters"
    
    return True, None

def validate_temperature(temperature: float) -> Tuple[bool, Optional[str]]:
    """
    Validate temperature parameter
    
    Args:
        temperature: Temperature value
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(temperature, (int, float)):
        return False, "temperature must be a number"
    
    if temperature < 0.1 or temperature > 2.0:
        return False, "temperature must be between 0.1 and 2.0"
    
    return True, None

def validate_custom_prompt(prompt: str) -> Tuple[bool, Optional[str]]:
    """
    Validate custom prompt
    
    Args:
        prompt: Custom prompt text
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(prompt, str):
        return False, "custom_prompt must be a string"
    
    if len(prompt.strip()) == 0:
        return False, "custom_prompt cannot be empty"
    
    if len(prompt) > 1000:
        return False, "custom_prompt cannot exceed 1000 characters"
    
    # Check for potentially harmful content
    harmful_patterns = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"eval\s*\(",
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False, "custom_prompt contains potentially harmful content"
    
    return True, None

def validate_batch_request(body: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate batch generation request
    
    Args:
        body: Request body
    
    Returns:
        Tuple of (is_valid, error_message, validated_body)
    """
    # Validate count
    count = body.get("count", 1)
    if not isinstance(count, int):
        return False, "count must be an integer", None
    
    if count < 1:
        return False, "count must be at least 1", None
    
    if count > MAX_BATCH_SIZE:
        return False, ERROR_BATCH_SIZE_EXCEEDED, None
    
    # Validate speech_types
    speech_types = body.get("speech_types", [])
    if not isinstance(speech_types, list):
        return False, "speech_types must be a list", None
    
    if len(speech_types) == 0:
        return False, "speech_types cannot be empty", None
    
    for speech_type in speech_types:
        is_valid, error = validate_speech_type(speech_type)
        if not is_valid:
            return False, f"Invalid speech_type in speech_types: {speech_type}", None
    
    # Return validated body with defaults
    validated_body = {
        "count": count,
        "speech_types": speech_types,
        "generation_type": body.get("generation_type", "speech"),
        "characters": body.get("characters", ["Hero", "Rival"]),
        "temperature": body.get("temperature", 0.8),
    }
    
    return True, None, validated_body

def sanitize_input(text: str) -> str:
    """
    Sanitize input text to prevent injection attacks
    
    Args:
        text: Input text to sanitize
    
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters (except newline, tab, carriage return)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit length
    if len(text) > 10000:
        text = text[:10000]
    
    return text.strip()

def validate_generation_request(body: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate single generation request
    
    Args:
        body: Request body
    
    Returns:
        Tuple of (is_valid, error_message, validated_body)
    """
    # Extract parameters with defaults
    speech_type = body.get("speech_type", "motivational")
    generation_type = body.get("type", "speech")
    custom_prompt = body.get("custom_prompt", "")
    characters = body.get("characters", ["Hero", "Rival"])
    temperature = body.get("temperature", 0.8)
    
    # Validate speech_type
    is_valid, error = validate_speech_type(speech_type)
    if not is_valid:
        return False, error, None
    
    # Validate generation_type
    is_valid, error = validate_generation_type(generation_type)
    if not is_valid:
        return False, error, None
    
    # Validate characters if generation_type is dialogue
    if generation_type == "dialogue":
        is_valid, error = validate_characters(characters)
        if not is_valid:
            return False, error, None
    
    # Validate custom_prompt if provided
    if custom_prompt:
        is_valid, error = validate_custom_prompt(custom_prompt)
        if not is_valid:
            return False, error, None
    
    # Validate temperature
    is_valid, error = validate_temperature(temperature)
    if not is_valid:
        return False, error, None
    
    # Sanitize inputs
    sanitized_custom_prompt = sanitize_input(custom_prompt)
    sanitized_characters = [sanitize_input(char) for char in characters]
    
    # Return validated body
    validated_body = {
        "speech_type": speech_type,
        "generation_type": generation_type,
        "custom_prompt": sanitized_custom_prompt,
        "characters": sanitized_characters,
        "temperature": temperature,
        "max_length": body.get("max_length", 200),
    }
    
    return True, None, validated_body

def validate_required_fields(body: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that required fields are present
    
    Args:
        body: Request body
        required_fields: List of required field names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    for field in required_fields:
        if field not in body:
            return False, ERROR_MISSING_REQUIRED_FIELD.format(field)
        
        value = body[field]
        if value is None or (isinstance(value, str) and len(value.strip()) == 0):
            return False, f"Field '{field}' cannot be empty"
    
    return True, None

def validate_json_body(body_str: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate JSON body string
    
    Args:
        body_str: JSON string
    
    Returns:
        Tuple of (is_valid, error_message, parsed_body)
    """
    if not body_str:
        return True, None, {}
    
    try:
        parsed_body = json.loads(body_str)
        if not isinstance(parsed_body, dict):
            return False, "Request body must be a JSON object", None
        
        return True, None, parsed_body
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}", None
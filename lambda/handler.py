"""
AWS Lambda Handler for Anime Quote / Speech Generator
------------------------------------------------------
Routes:
  GET  /health          -> health check
  POST /generate        -> generate a speech or dialogue
  POST /generate/batch  -> generate multiple speeches

Event format (API Gateway HTTP API v2 payload):
  {
    "requestContext": { "http": { "method": "POST" } },
    "rawPath": "/generate",
    "body": "<JSON string>"
  }
"""

import json
import os
import random
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Speech prompts used when Gemini API is unavailable (fallback mode)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Gemini / CrewAI generation
# ---------------------------------------------------------------------------

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
    # Import here so Lambda cold-start is faster when running in fallback mode
    from crewai import Agent, Task, Crew, LLM  # noqa: PLC0415

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


def _generate_fallback(speech_type: str) -> str:
    """Return a pre-written fallback quote when Gemini is unavailable."""
    pool = FALLBACK_PROMPTS.get(speech_type, FALLBACK_PROMPTS["motivational"])
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Core generation orchestrator
# ---------------------------------------------------------------------------

def generate_content(body: dict) -> dict:
    """
    Parameters (all optional):
      speech_type     : motivational | battle | friendship | determination | villain
      type            : speech | dialogue
      custom_prompt   : free-form string
      characters      : list of character names (used for dialogue)
      temperature     : float 0.5-1.0 (passed as hint; Gemini handles internally)
    """
    speech_type = body.get("speech_type", "motivational")
    generation_type = body.get("type", "speech")
    custom_prompt = body.get("custom_prompt", "")
    characters = body.get("characters", ["Hero", "Rival"])

    logger.info(
        "Generating content | type=%s | speech_type=%s | custom_prompt=%s",
        generation_type, speech_type, bool(custom_prompt),
    )

    use_fallback = False
    try:
        content = _generate_with_gemini(generation_type, speech_type, custom_prompt, characters)
    except ImportError:
        logger.warning("crewai not installed — using fallback quotes")
        use_fallback = True
        content = _generate_fallback(speech_type)
    except ValueError as exc:
        logger.warning("Configuration error (%s) — using fallback quotes", exc)
        use_fallback = True
        content = _generate_fallback(speech_type)
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini call failed: %s", exc)
        use_fallback = True
        content = _generate_fallback(speech_type)

    return {
        "speech_type": speech_type,
        "generation_type": generation_type,
        "content": content,
        "fallback_used": use_fallback,
    }


def generate_batch(body: dict) -> dict:
    """Generate multiple speeches in one Lambda call."""
    count = min(int(body.get("count", 3)), 10)
    speech_types = body.get(
        "speech_types", ["motivational", "battle", "friendship", "determination", "villain"]
    )

    results = []
    for i in range(count):
        speech_type = speech_types[i % len(speech_types)]
        result = generate_content({"speech_type": speech_type, "type": "speech"})
        results.append(result)

    return {"count": count, "speeches": results}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _cors_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }


def _ok(body: dict) -> dict:
    return {"statusCode": 200, "headers": _cors_headers(), "body": json.dumps(body)}


def _err(status: int, message: str) -> dict:
    return {
        "statusCode": status,
        "headers": _cors_headers(),
        "body": json.dumps({"error": message}),
    }


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event, context):  # noqa: ANN001
    """AWS Lambda entry point."""
    logger.info("Event: %s", json.dumps(event))

    # Handle CORS pre-flight
    http_ctx = event.get("requestContext", {}).get("http", {})
    method = http_ctx.get("method", event.get("httpMethod", "GET")).upper()
    path = event.get("rawPath", event.get("path", "/"))

    if method == "OPTIONS":
        return _ok({"message": "CORS preflight OK"})

    # ------- Routes -------

    if path in ("/health", "/health/") and method == "GET":
        return _ok({
            "status": "healthy",
            "service": "anime-quote-generator",
            "gemini_configured": bool(os.environ.get("GOOGLE_API_KEY")),
        })

    if path in ("/generate", "/generate/") and method == "POST":
        try:
            body = json.loads(event.get("body") or "{}")
        except json.JSONDecodeError:
            return _err(400, "Invalid JSON body")

        result = generate_content(body)
        return _ok(result)

    if path in ("/generate/batch", "/generate/batch/") and method == "POST":
        try:
            body = json.loads(event.get("body") or "{}")
        except json.JSONDecodeError:
            return _err(400, "Invalid JSON body")

        result = generate_batch(body)
        return _ok(result)

    return _err(404, f"Route not found: {method} {path}")

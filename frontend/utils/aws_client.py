"""
API client for the Anime Quote Generator Streamlit dashboard.

Communicates with the FastAPI backend and provides fallback
simulation mode when the backend is unavailable.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = os.getenv("ANIME_QUOTE_API_URL", "http://localhost:8000/api/v1")
DEFAULT_API_KEY = os.getenv("ANIME_QUOTE_API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("ANIME_QUOTE_REQUEST_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_auth_token() -> Optional[str]:
    """Retrieve the JWT token from Streamlit session state."""
    return st.session_state.get("auth_token")


def _get_api_key() -> str:
    """Retrieve the API key from session state or environment."""
    return st.session_state.get("api_key", DEFAULT_API_KEY)


def _headers(json_content: bool = True) -> Dict[str, str]:
    """Build request headers with optional JWT and API key."""
    hdrs: Dict[str, str] = {}
    if json_content:
        hdrs["Content-Type"] = "application/json"
    token = _get_auth_token()
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    api_key = _get_api_key()
    if api_key:
        hdrs["X-API-Key"] = api_key
    return hdrs


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _api_url(path: str) -> str:
    """Build a full API URL from a relative path."""
    base = st.session_state.get("api_base_url", DEFAULT_API_BASE)
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _handle_response(response: requests.Response) -> Dict[str, Any]:
    """Process an API response, raising on errors."""
    if response.status_code == 200 or response.status_code == 201 or response.status_code == 202:
        return response.json()
    if response.status_code == 401:
        st.session_state.pop("auth_token", None)
        st.error("Session expired. Please log in again.")
        return {"error": "unauthorized", "detail": "Session expired"}
    if response.status_code == 429:
        return {"error": "rate_limited", "detail": "Rate limit exceeded. Please try again later."}
    try:
        error_data = response.json()
        return {"error": f"http_{response.status_code}", "detail": error_data}
    except Exception:
        return {"error": f"http_{response.status_code}", "detail": response.text}


# ---------------------------------------------------------------------------
# Public API methods
# ---------------------------------------------------------------------------

def check_health() -> Dict[str, Any]:
    """Check backend health status."""
    try:
        resp = requests.get(_api_url("health"), timeout=10)
        return _handle_response(resp)
    except requests.ConnectionError:
        return {"status": "unavailable", "backend": "offline"}
    except requests.Timeout:
        return {"status": "timeout", "backend": "slow"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def login(username: str, password: str) -> Dict[str, Any]:
    """Authenticate with the backend and store the JWT token."""
    try:
        resp = requests.post(
            _api_url("auth/login"),
            json={"username": username, "password": password},
            timeout=REQUEST_TIMEOUT,
        )
        result = _handle_response(resp)
        if "access_token" in result:
            st.session_state["auth_token"] = result["access_token"]
            st.session_state["refresh_token"] = result.get("refresh_token")
            st.session_state["user_info"] = result.get("user", {})
            st.session_state["authenticated"] = True
        return result
    except requests.ConnectionError:
        return _simulate_login(username, password)
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def register(username: str, email: str, password: str) -> Dict[str, Any]:
    """Register a new user account."""
    try:
        resp = requests.post(
            _api_url("auth/register"),
            json={"username": username, "email": email, "password": password},
            timeout=REQUEST_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_register(username, email)
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def logout() -> None:
    """Clear authentication state."""
    st.session_state.pop("auth_token", None)
    st.session_state.pop("refresh_token", None)
    st.session_state.pop("user_info", None)
    st.session_state.pop("authenticated", None)


def get_current_user() -> Dict[str, Any]:
    """Get the current authenticated user's profile."""
    try:
        resp = requests.get(_api_url("auth/me"), headers=_headers(), timeout=REQUEST_TIMEOUT)
        return _handle_response(resp)
    except requests.ConnectionError:
        return st.session_state.get("user_info", {})
    except Exception as e:
        return {"error": str(e)}


def submit_generation(
    speech_type: str = "motivational",
    generation_type: str = "quote",
    custom_prompt: str = "",
    characters: Optional[List[str]] = None,
    temperature: float = 0.8,
    max_length: int = 200,
) -> Dict[str, Any]:
    """Submit a single generation request."""
    payload = {
        "speech_type": speech_type,
        "generation_type": generation_type,
        "custom_prompt": custom_prompt,
        "characters": characters or [],
        "temperature": temperature,
        "max_length": max_length,
    }
    try:
        resp = requests.post(
            _api_url("generate"),
            json=payload,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_generation(payload)
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def submit_batch_generation(
    speech_types: List[str],
    generation_type: str = "quote",
    count: int = 5,
    temperature: float = 0.8,
    max_length: int = 200,
) -> Dict[str, Any]:
    """Submit a batch generation request."""
    payload = {
        "speech_types": speech_types,
        "generation_type": generation_type,
        "count": count,
        "temperature": temperature,
        "max_length": max_length,
    }
    try:
        resp = requests.post(
            _api_url("generate/batch"),
            json=payload,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_batch_generation(payload)
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a generation job."""
    try:
        resp = requests.get(
            _api_url(f"generate/{job_id}/status"),
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_job_status(job_id)
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def get_quotes(
    page: int = 1,
    page_size: int = 20,
    speech_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a paginated list of generated quotes."""
    params: Dict[str, Any] = {"page": page, "page_size": page_size}
    if speech_type:
        params["speech_type"] = speech_type
    try:
        resp = requests.get(
            _api_url("quotes"),
            params=params,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_quotes(page, page_size, speech_type)
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def get_random_quote() -> Dict[str, Any]:
    """Get a random generated quote."""
    try:
        resp = requests.get(_api_url("quotes/random"), timeout=REQUEST_TIMEOUT)
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_random_quote()
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def get_stats() -> Dict[str, Any]:
    """Get pipeline statistics."""
    try:
        resp = requests.get(_api_url("stats"), headers=_headers(), timeout=REQUEST_TIMEOUT)
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_stats()
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def get_config() -> Dict[str, Any]:
    """Get pipeline configuration (admin only)."""
    try:
        resp = requests.get(_api_url("config"), headers=_headers(), timeout=REQUEST_TIMEOUT)
        return _handle_response(resp)
    except requests.ConnectionError:
        return _simulate_config()
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


# ---------------------------------------------------------------------------
# Simulation / Fallback Mode
# ---------------------------------------------------------------------------

_SIMULATION_QUOTES = [
    {
        "id": "sim-001",
        "text": "Even in the darkest night, the stars still shine. That is the will of the warrior!",
        "speech_type": "motivational",
        "character": "Hero",
        "source": "simulation",
        "created_at": "2025-01-15T10:30:00Z",
    },
    {
        "id": "sim-002",
        "text": "Power without purpose is merely destruction. I choose the path of meaning.",
        "speech_type": "philosophical",
        "character": "Sage",
        "source": "simulation",
        "created_at": "2025-01-15T11:00:00Z",
    },
    {
        "id": "sim-003",
        "text": "You think you've seen my full power? This is just the beginning!",
        "speech_type": "villain",
        "character": "Dark Lord",
        "source": "simulation",
        "created_at": "2025-01-15T11:30:00Z",
    },
    {
        "id": "sim-004",
        "text": "I made a promise, and I intend to keep it. That's what it means to be a hero!",
        "speech_type": "heroic",
        "character": "Champion",
        "source": "simulation",
        "created_at": "2025-01-15T12:00:00Z",
    },
    {
        "id": "sim-005",
        "text": "The cherry blossoms fall, but they return each spring. Such is the cycle of hope.",
        "speech_type": "emotional",
        "character": "Poet",
        "source": "simulation",
        "created_at": "2025-01-15T12:30:00Z",
    },
    {
        "id": "sim-006",
        "text": "In the silence between heartbeats, you'll find the answer you've been seeking.",
        "speech_type": "philosophical",
        "character": "Monk",
        "source": "simulation",
        "created_at": "2025-01-15T13:00:00Z",
    },
    {
        "id": "sim-007",
        "text": "Comedy is just tragedy plus time, and I've had centuries to laugh at your mistakes!",
        "speech_type": "comedic",
        "character": "Trickster",
        "source": "simulation",
        "created_at": "2025-01-15T13:30:00Z",
    },
    {
        "id": "sim-008",
        "text": "I will not falter. I will not yield. For everyone who believed in me, I will stand firm!",
        "speech_type": "motivational",
        "character": "Guardian",
        "source": "simulation",
        "created_at": "2025-01-15T14:00:00Z",
    },
]

_SIMULATION_JOBS: Dict[str, Dict[str, Any]] = {}


def _simulate_login(username: str, password: str) -> Dict[str, Any]:
    """Simulate login when backend is unavailable."""
    if password:
        token = f"sim_token_{username}_{int(time.time())}"
        st.session_state["auth_token"] = token
        st.session_state["authenticated"] = True
        st.session_state["user_info"] = {
            "id": f"sim-{username}",
            "username": username,
            "email": f"{username}@example.com",
            "role": "admin" if username == "admin" else "user",
        }
        return {
            "access_token": token,
            "refresh_token": f"sim_refresh_{username}",
            "token_type": "bearer",
            "user": st.session_state["user_info"],
        }
    return {"error": "invalid_credentials", "detail": "Invalid username or password"}


def _simulate_register(username: str, email: str) -> Dict[str, Any]:
    """Simulate registration when backend is unavailable."""
    return {
        "id": f"sim-{username}",
        "username": username,
        "email": email,
        "role": "user",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _simulate_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a generation request when backend is unavailable."""
    import uuid

    job_id = str(uuid.uuid4())
    _SIMULATION_JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "speech_type": payload.get("speech_type", "motivational"),
        "generation_type": payload.get("generation_type", "quote"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload": payload,
    }
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Generation request submitted (simulation mode)",
        "estimated_time": "5-10 seconds",
    }


def _simulate_batch_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a batch generation request."""
    import uuid

    batch_id = str(uuid.uuid4())
    jobs = []
    for speech_type in payload.get("speech_types", ["motivational"]):
        for _ in range(payload.get("count", 5)):
            job_id = str(uuid.uuid4())
            _SIMULATION_JOBS[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "speech_type": speech_type,
                "generation_type": payload.get("generation_type", "quote"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            jobs.append(job_id)
    return {
        "batch_id": batch_id,
        "job_ids": jobs,
        "total_jobs": len(jobs),
        "status": "pending",
        "message": "Batch generation request submitted (simulation mode)",
    }


def _simulate_job_status(job_id: str) -> Dict[str, Any]:
    """Simulate job status lookup."""
    if job_id in _SIMULATION_JOBS:
        job = _SIMULATION_JOBS[job_id]
        # Simulate progression
        elapsed = time.time() - time.mktime(
            time.strptime(job["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        )
        if elapsed > 8:
            job["status"] = "completed"
            matching = [q for q in _SIMULATION_QUOTES if q["speech_type"] == job["speech_type"]]
            job["result"] = matching[0] if matching else _SIMULATION_QUOTES[0]
        elif elapsed > 3:
            job["status"] = "generating"
        elif elapsed > 1:
            job["status"] = "preprocessing"
        return job
    return {"error": "not_found", "detail": f"Job {job_id} not found"}


def _simulate_quotes(page: int, page_size: int, speech_type: Optional[str]) -> Dict[str, Any]:
    """Simulate a paginated quotes response."""
    filtered = _SIMULATION_QUOTES
    if speech_type:
        filtered = [q for q in filtered if q["speech_type"] == speech_type]
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "quotes": filtered[start:end],
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (len(filtered) + page_size - 1) // page_size),
    }


def _simulate_random_quote() -> Dict[str, Any]:
    """Simulate a random quote response."""
    import random
    return random.choice(_SIMULATION_QUOTES)


def _simulate_stats() -> Dict[str, Any]:
    """Simulate pipeline statistics."""
    return {
        "total_generations": 1247,
        "successful_generations": 1189,
        "failed_generations": 58,
        "success_rate": 95.3,
        "average_processing_time": 3.2,
        "active_jobs": 3,
        "queue_depth": 7,
        "dlq_count": 2,
        "generations_by_type": {
            "motivational": 342,
            "villain": 278,
            "philosophical": 215,
            "heroic": 189,
            "emotional": 134,
            "comedic": 89,
        },
        "generations_by_source": {
            "gemini": 678,
            "gpt2": 389,
            "fallback": 122,
        },
        "daily_generations": [
            {"date": f"2025-01-{d:02d}", "count": 30 + (d % 20) * 3}
            for d in range(1, 16)
        ],
        "hourly_generations": [
            {"hour": h, "count": 5 + (h % 12) * 2}
            for h in range(24)
        ],
        "cost_estimate": {
            "lambda_cost": 2.45,
            "s3_cost": 0.12,
            "sqs_cost": 0.03,
            "sns_cost": 0.01,
            "dynamodb_cost": 0.89,
            "api_gateway_cost": 1.23,
            "total": 4.73,
        },
    }


def _simulate_config() -> Dict[str, Any]:
    """Simulate pipeline configuration."""
    return {
        "environment": "development",
        "region": "us-east-1",
        "api_gateway_url": "https://api.example.com/v1",
        "lambda_functions": {
            "preprocessing": {"memory": 256, "timeout": 30},
            "generation": {"memory": 1024, "timeout": 120},
            "postprocessing": {"memory": 256, "timeout": 60},
        },
        "s3_bucket": "anime-quote-generator-dev",
        "sqs_queues": {
            "generation_queue": {"visibility_timeout": 120, "max_receive_count": 3},
            "postprocessing_queue": {"visibility_timeout": 60, "max_receive_count": 3},
        },
        "sns_topics": ["generation-complete", "generation-failed", "system-alerts"],
        "dynamodb_tables": ["jobs", "history", "metrics"],
        "rate_limits": {
            "requests_per_second": 10,
            "burst_limit": 50,
        },
    }

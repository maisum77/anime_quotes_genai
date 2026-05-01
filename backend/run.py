"""
Run Script
----------
Launch the FastAPI application with uvicorn.

Usage:
    python run.py                  # Production mode
    python run.py --reload         # Development mode with auto-reload
    python run.py --port 8080      # Custom port
"""

import argparse
import uvicorn

from app.core.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anime Quote Generator FastAPI Backend"
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"Host to bind to (default: {settings.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind to (default: {settings.port})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.reload,
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.lower(),
        help=f"Log level (default: {settings.log_level})",
    )

    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

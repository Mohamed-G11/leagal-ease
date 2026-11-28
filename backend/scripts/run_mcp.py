"""Start the MCP server (see app/mcp_server.py). Works from any working directory.

    python backend/scripts/run_mcp.py [--http]

Settings and secrets come from the repository's .env file, so no client configuration needs to hold a key.
Variables already in the environment (docker compose) win over the file.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare_environment() -> None:
    from dotenv import load_dotenv

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/, so that `app` can be imported
    load_dotenv(REPO_ROOT / ".env")  # in Docker there is no such file
    os.environ.setdefault("PGCONNECT_TIMEOUT", "3")  # logging a request must not hang when PostgreSQL is off
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


if __name__ == "__main__":
    prepare_environment()
    from app.mcp_server import main

    main()

"""LLM layer: query interpretation and explanation generation."""

import os
from pathlib import Path

MODEL = "claude-opus-5"


def _load_env() -> None:
    """Load KEY=VALUE lines from the repo-root .env (if present) into the
    environment, without overriding variables that are already set."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env()

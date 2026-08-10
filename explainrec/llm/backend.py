"""LLM backends: Anthropic API or the local Claude Code CLI.

Both backends expose the same two calls:

- ``text(system, user) -> str``
- ``structured(system, user, schema) -> schema instance``

``ApiBackend`` uses the Anthropic SDK (needs ``ANTHROPIC_API_KEY``;
structured outputs enforce the schema server-side). ``CliBackend``
shells out to a local ``claude -p`` (Claude Code CLI, billed to the
subscription, no API key needed); the schema is enforced client-side by
prompting for JSON and validating with Pydantic, with one retry that
feeds the validation error back.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from . import MODEL

M = TypeVar("M", bound=BaseModel)


class Backend(Protocol):
    def text(self, system: str, user: str) -> str: ...
    def structured(self, system: str, user: str, schema: type[M]) -> M: ...


class ApiBackend:
    def __init__(self, client=None, model: str = MODEL) -> None:
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model

    def text(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        return "".join(b.text for b in response.content if b.type == "text")

    def structured(self, system: str, user: str, schema: type[M]) -> M:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        if response.parsed_output is None:
            raise RuntimeError(f"could not parse a {schema.__name__} from the model output")
        return response.parsed_output


class CliBackend:
    """Non-interactive local Claude Code CLI (``claude -p``).

    Tools and MCP servers are disabled so the call is a plain completion.
    ``model`` takes CLI aliases ("opus", "sonnet", "haiku") or full ids.
    """

    def __init__(self, model: str = "opus", binary: str = "claude",
                 timeout: int = 600) -> None:
        self.model = model
        self.binary = binary
        self.timeout = timeout

    @staticmethod
    def _clean_env() -> dict[str, str]:
        """Subprocess env without API-auth or nested-session variables.

        The CLI prefers ANTHROPIC_API_KEY (which our .env loader may have
        exported) over the subscription login, and refuses to run cleanly
        inside another Claude Code session; strip both classes of variable
        so the call always uses the local `claude` login.
        """
        drop = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")
        return {
            k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith("CLAUDE_")
        }

    def text(self, system: str, user: str) -> str:
        cmd = [
            self.binary, "-p",
            "--tools", "",
            "--strict-mcp-config",
            "--model", self.model,
            "--system-prompt", system,
        ]
        try:
            result = subprocess.run(
                cmd, input=user, capture_output=True, text=True,
                timeout=self.timeout, env=self._clean_env(),
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{self.binary!r} not found; install Claude Code or use the API backend"
            ) from e
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr.strip()[:500]}")
        return result.stdout.strip()

    def structured(self, system: str, user: str, schema: type[M]) -> M:
        schema_json = json.dumps(schema.model_json_schema())
        system_full = (
            f"{system}\n\n"
            f"Respond with a single JSON object that validates against this JSON "
            f"schema, and nothing else (no prose, no code fences):\n{schema_json}"
        )
        raw = self.text(system_full, user)
        try:
            return schema.model_validate_json(extract_json(raw))
        except (ValidationError, ValueError) as first_error:
            retry_user = (
                f"{user}\n\nYour previous reply was not valid:\n{raw}\n\n"
                f"Error: {first_error}\n"
                f"Reply again with only a valid JSON object."
            )
            raw = self.text(system_full, retry_user)
            return schema.model_validate_json(extract_json(raw))


def extract_json(text: str) -> str:
    """Pull the outermost JSON object out of a possibly fenced/prosy reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found in reply: {text[:200]!r}")
    return text[start : end + 1]


def get_backend(name: str, **kwargs) -> Backend:
    if name == "api":
        return ApiBackend(**kwargs)
    if name == "cli":
        return CliBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; expected 'api' or 'cli'")

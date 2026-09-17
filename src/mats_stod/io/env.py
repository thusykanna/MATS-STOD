"""Loading credentials from a .env file.

Credentials never live in YAML, because configs are committed and copied into
every run directory. They come from the environment instead, and a gitignored
`.env` at the repository root is a convenient way to set that environment once
rather than exporting in every new terminal.

A variable already present in the real environment always wins, so an explicit
`export` still overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import REPO_ROOT

#: Only these may be set from .env. An allowlist, so a stray line in the file
#: cannot quietly change unrelated behaviour such as PATH or PYTHONPATH.
ALLOWED = (
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_API_KEY",
)


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Load `.env` into the environment. Returns the names that were set.

    Silently does nothing when the file is absent, which is the normal case in
    tests and in CI.
    """
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return []

    applied: list[str] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in ALLOWED:
            continue
        # The real environment wins, so an explicit export still overrides.
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


def credential_status() -> dict[str, str]:
    """What is set right now, with secrets reduced to present or absent.

    Used by `mats-stod check-llm` so a misconfiguration is diagnosable without
    printing a key into a terminal or a screenshot.
    """
    out: dict[str, str] = {}
    for name in ALLOWED:
        value = os.environ.get(name)
        if not value:
            out[name] = "not set"
        elif "KEY" in name or "CREDENTIALS" in name:
            out[name] = f"set ({len(value)} chars)"
        else:
            out[name] = value
    return out

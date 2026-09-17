"""Prompt loading.

Prompts are versioned files, never inline strings, because the prompt is an
experimental variable: a result is only reproducible if the exact prompt text
that produced it can be recovered from the run's config and the repository.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,  # a missing slot is a bug, not an empty string
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render(version: str, **slots: object) -> str:
    """Render a versioned prompt template by name, e.g. "translate_v1"."""
    template = _env().get_template(f"{version}.jinja")
    return template.render(**slots).strip()


def prompt_text(version: str) -> str:
    """The raw template source, for archiving into a run directory."""
    return (PROMPT_DIR / f"{version}.jinja").read_text(encoding="utf-8")


def available() -> list[str]:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.jinja"))

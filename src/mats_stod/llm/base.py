"""Provider-neutral LLM interface.

Only modules inside this package may import a vendor SDK. Everything else in
MATS-STOD depends on `LLMClient`, so swapping the model, which is still an open
research decision, never touches the pipeline.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    model: str = ""
    latency_s: float = 0.0
    parsed: Any = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Provider call failed after its own retries."""


class ParseError(ValueError):
    """The model's output did not satisfy the requested schema."""


class LLMClient(ABC):
    """One call, one response. Caching and accounting wrap this."""

    #: Name recorded in cache keys and TranslationRecords.
    model: str = "unknown"
    provider: str = "unknown"

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        **params: Any,
    ) -> LLMResponse:
        """Generate one completion.

        `schema` is a JSON schema. Providers that support structured output
        enforce it; providers that do not fall back to strict parsing.
        """


def system_user(system: str, user: str) -> list[Message]:
    return [Message("system", system), Message("user", user)]


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str, key: str | None = None) -> Any:
    """Parse JSON from a model response, tolerating fenced code blocks.

    Models wrap JSON in markdown fences often enough that failing on it would
    waste a retry on a response that is actually correct.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(candidate)
        if not match:
            raise ParseError(f"no JSON object in response: {text[:200]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON in response: {text[:200]!r}") from exc
    if key is not None:
        if not isinstance(data, dict) or key not in data:
            raise ParseError(f"response JSON lacks key {key!r}: {text[:200]!r}")
        return data[key]
    return data


def parse_yes_no(text: str) -> bool:
    """Parse GRAFT's one-token yes/no answers.

    Returns True only on an explicit yes. Anything else, including an empty or
    truncated response, is a boundary or a missing edge: the conservative
    reading, and the one the caller counts as `unparsed`.
    """
    token = text.strip().lower()
    if not token:
        raise ParseError("empty yes/no response")
    if token.startswith("yes") or token in {"y", "true", "1"}:
        return True
    if token.startswith("no") or token in {"n", "false", "0"}:
        return False
    raise ParseError(f"not a yes/no answer: {text[:80]!r}")

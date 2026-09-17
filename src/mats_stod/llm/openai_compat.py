"""Optional OpenAI-compatible provider.

Kept so that model choice, which is still an open research decision, can be
compared against a non-Google model without touching the pipeline. Not used by
the default config.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..config import LLMSettings
from .base import LLMClient, LLMError, LLMResponse, Message


class OpenAICompatClient(LLMClient):
    """Any endpoint that speaks the OpenAI chat-completions API."""

    provider = "openai_compat"

    def __init__(self, settings: LLMSettings) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "openai is not installed. Install the optional dependency:\n"
                "  uv sync --extra openai"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise LLMError("OPENAI_API_KEY (or LLM_API_KEY) is not set.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.settings = settings
        self.model = settings.model

    def complete(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        **params: Any,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": params.get("temperature", self.settings.temperature),
            "max_tokens": params.get("max_output_tokens", self.settings.max_output_tokens),
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if params.get("stop"):
            kwargs["stop"] = list(params["stop"])

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                started = time.perf_counter()
                resp = self._client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - started
                usage = resp.usage
                return LLMResponse(
                    text=(resp.choices[0].message.content or "").strip(),
                    tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
                    tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
                    model=self.model,
                    latency_s=latency,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.settings.max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise LLMError(f"OpenAI-compatible call failed: {last_error}")

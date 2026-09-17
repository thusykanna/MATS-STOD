"""Google Gemini provider.

The only module in MATS-STOD that imports a Google SDK. It serves two
backends behind one class: Vertex AI (the default, using free-trial credits on
a GCP project) and AI Studio (an API key, kept as a fallback). Both speak the
same `google-genai` client, so switching is a config change and produces
identical cache keys for identical prompts.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..config import LLMSettings
from .base import LLMClient, LLMError, LLMResponse, Message


def _import_genai():  # pragma: no cover - exercised only with the SDK installed
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise LLMError(
            "google-genai is not installed. Install the optional dependency:\n"
            "  uv sync --extra gemini"
        ) from exc
    return genai, types


class GeminiClient(LLMClient):
    """Gemini through Vertex AI or AI Studio."""

    provider = "gemini"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.model = settings.model
        self.backend = settings.backend
        genai, types = _import_genai()
        self._types = types

        if settings.backend == "vertex":
            project = settings.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION") or settings.location
            if not project:
                raise LLMError(
                    "Vertex backend needs a project. Set GOOGLE_CLOUD_PROJECT or "
                    "llm.project in the config, and run "
                    "`gcloud auth application-default login`."
                )
            self._client = genai.Client(vertexai=True, project=project, location=location)
            self.provider = "gemini-vertex"
        elif settings.backend == "ai_studio":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise LLMError("AI Studio backend needs GEMINI_API_KEY in the environment.")
            self._client = genai.Client(api_key=api_key)
            self.provider = "gemini-aistudio"
        else:
            raise LLMError(f"unknown Gemini backend: {settings.backend!r}")

    def _config(self, schema: dict[str, Any] | None, params: dict[str, Any]) -> Any:
        types = self._types
        kwargs: dict[str, Any] = {
            "temperature": params.get("temperature", self.settings.temperature),
            "max_output_tokens": params.get(
                "max_output_tokens", self.settings.max_output_tokens
            ),
        }
        if self.settings.top_p is not None:
            kwargs["top_p"] = self.settings.top_p
        if "system" in params and params["system"]:
            kwargs["system_instruction"] = params["system"]
        if schema is not None:
            # Structured output: the model is constrained rather than asked
            # politely, which removes most parse-failure retries.
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        if params.get("stop"):
            kwargs["stop_sequences"] = list(params["stop"])
        return types.GenerateContentConfig(**kwargs)

    def complete(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        **params: Any,
    ) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        body = "\n\n".join(m.content for m in messages if m.role != "system")
        call_params = dict(params)
        if system:
            call_params["system"] = system

        config = self._config(schema, call_params)
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                started = time.perf_counter()
                resp = self._client.models.generate_content(
                    model=self.model, contents=body, config=config
                )
                latency = time.perf_counter() - started
                usage = getattr(resp, "usage_metadata", None)
                return LLMResponse(
                    text=(resp.text or "").strip(),
                    tokens_in=int(getattr(usage, "prompt_token_count", 0) or 0),
                    tokens_out=int(getattr(usage, "candidates_token_count", 0) or 0),
                    cached=False,
                    model=self.model,
                    latency_s=latency,
                )
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = exc
                if attempt == self.settings.max_retries - 1:
                    break
                # Exponential backoff: free-tier quota errors are the common
                # failure and they clear with time, not with a fast retry.
                time.sleep(2**attempt)
        raise LLMError(
            f"Gemini call failed after {self.settings.max_retries} attempts: {last_error}"
        )

    def count_tokens(self, text: str) -> int:
        """Ask the provider to count tokens; used for budgeting only."""
        try:
            result = self._client.models.count_tokens(model=self.model, contents=text)
            return int(getattr(result, "total_tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"token counting failed: {exc}") from exc

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


#: Substrings identifying errors that will fail identically on every retry.
#: A quota error (429) is deliberately absent: that one does clear with time.
_PERMANENT = (
    "PERMISSION_DENIED",
    "NOT_FOUND",
    "INVALID_ARGUMENT",
    "UNAUTHENTICATED",
    "SERVICE_DISABLED",
)

#: Raw API errors are long JSON blobs. These map the diagnostic part to the
#: one action that fixes it, so a setup mistake does not look like a bug.
_HINTS: tuple[tuple[str, str], ...] = (
    (
        "SERVICE_DISABLED",
        "The Vertex AI API is not enabled on this project. Run:\n"
        "  gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID",
    ),
    (
        "was not found or your project does not have access",
        "This model name is not available to your project in this region. Model names\n"
        "  change, and a model can be listed without being callable. Try another, for\n"
        "  example llm.model: gemini-2.5-flash, or another llm.location.",
    ),
    (
        "PERMISSION_DENIED",
        "The credentials cannot use this project. Check that GOOGLE_CLOUD_PROJECT names\n"
        "  a real project you can access, that billing is enabled on it, and that you\n"
        "  hold the Vertex AI User role. Then run:\n"
        "  gcloud auth application-default login",
    ),
    (
        "UNAUTHENTICATED",
        "No usable credentials. Run:\n  gcloud auth application-default login",
    ),
    (
        "RESOURCE_EXHAUSTED",
        "Quota exceeded. Wait and retry, lower --max-docs, or request more quota.",
    ),
    (
        "could not automatically determine credentials",
        "Application Default Credentials are missing. Run:\n"
        "  gcloud auth application-default login",
    ),
)


def _hit_token_cap(response: Any) -> bool:
    """Whether generation stopped because it ran out of output tokens."""
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return False
    return "MAX_TOKENS" in str(reason).upper()


def kwargs_cap(params: dict[str, Any], settings: LLMSettings) -> int:
    """The output-token cap that applied to one call, for error messages."""
    return int(params.get("max_output_tokens", settings.max_output_tokens))


def is_permanent_error(exc: Exception | None) -> bool:
    """Whether retrying this error could ever produce a different result."""
    text = str(exc)
    return any(marker in text for marker in _PERMANENT)


def explain_error(exc: Exception | None) -> str:
    """Attach the fix to a raw provider error, keeping the original text."""
    raw = str(exc)
    for marker, hint in _HINTS:
        if marker in raw:
            return f"{raw}\n\nWhat to do:\n  {hint}"
    return raw


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

        # Gemini 2.5 and later reason before answering, and those thinking
        # tokens are charged against max_output_tokens. A binary yes/no call
        # with a small cap therefore returns nothing at all unless thinking is
        # switched off, so the caller can disable it per call.
        budget = params.get("thinking_budget", self.settings.thinking_budget)
        if budget is not None:
            kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=budget)

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
                text = (resp.text or "").strip()
                thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)

                # An empty answer that stopped at the token cap is a
                # configuration fault, not a decision. Returning it would let a
                # yes/no caller read silence as "no" and quietly produce a
                # degenerate segmentation, so it is raised instead.
                if not text and _hit_token_cap(resp):
                    raise LLMError(
                        f"the model returned no text: it reached max_output_tokens "
                        f"({kwargs_cap(params, self.settings)}) "
                        f"after spending {thoughts} tokens on internal reasoning.\n\n"
                        "What to do:\n"
                        "  Thinking tokens count against max_output_tokens on Gemini 2.5\n"
                        "  and later. Pass thinking_budget=0 for short classification\n"
                        "  calls, or raise max_output_tokens."
                    )

                return LLMResponse(
                    text=text,
                    tokens_in=int(getattr(usage, "prompt_token_count", 0) or 0),
                    tokens_out=int(getattr(usage, "candidates_token_count", 0) or 0),
                    cached=False,
                    model=self.model,
                    latency_s=latency,
                    raw={"thoughts_tokens": thoughts},
                )
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = exc
                if is_permanent_error(exc):
                    # A misconfigured project, a retired model or a missing
                    # permission will fail identically every time. Retrying
                    # only delays a message the user needs immediately.
                    raise LLMError(explain_error(exc)) from exc
                if attempt == self.settings.max_retries - 1:
                    break
                # Exponential backoff: quota errors are the common transient
                # failure and they clear with time, not with a fast retry.
                time.sleep(2**attempt)
        raise LLMError(
            f"failed after {self.settings.max_retries} attempts. {explain_error(last_error)}"
        )

    def count_tokens(self, text: str) -> int:
        """Ask the provider to count tokens; used for budgeting only."""
        try:
            result = self._client.models.count_tokens(model=self.model, contents=text)
            return int(getattr(result, "total_tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"token counting failed: {exc}") from exc

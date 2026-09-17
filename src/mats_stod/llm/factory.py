"""Builds the LLM client a run uses, from config."""

from __future__ import annotations

from ..config import Settings
from .base import LLMClient, LLMError
from .cache import LLMCache
from .client import CachedLLM
from .cost import CostLedger
from .fake import EchoLLM


def build_provider(settings: Settings, provider_override: str | None = None) -> LLMClient:
    """Instantiate the raw provider.

    Vendor SDKs are imported inside the branch that needs them so that a run
    with the fake provider works without any cloud dependency installed.
    """
    name = provider_override or settings.llm.provider
    if name == "fake":
        # Tests inject their own FakeLLM; this one only exists so a
        # command can be smoke-tested offline.
        return EchoLLM()
    if name == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(settings.llm)
    if name == "openai_compat":
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(settings.llm)
    raise LLMError(f"unknown provider: {name!r}")


def build_llm(
    settings: Settings,
    provider: LLMClient | None = None,
    provider_override: str | None = None,
    dry_run: bool = False,
) -> CachedLLM:
    """Wrap a provider with the cache and the ledger.

    Tests pass `provider` directly (a FakeLLM); commands let the factory build
    one from config.
    """
    raw = provider if provider is not None else build_provider(settings, provider_override)
    cache = LLMCache(settings.llm.cache_path) if settings.llm.use_cache else None
    ledger = CostLedger(prices_usd_per_mtok=settings.llm.prices_usd_per_mtok)
    return CachedLLM(raw, cache, ledger, dry_run=dry_run)

"""The client every pipeline actually uses.

`CachedLLM` wraps a provider with the disk cache, the call ledger and a
dry-run mode. Pipelines hold one of these and never see a provider directly,
which is what keeps the budget controls in a single place.
"""

from __future__ import annotations

import time
from typing import Any

from .base import LLMClient, LLMResponse, Message
from .cache import LLMCache, cache_key
from .ledger import CallLedger


class DryRunExhausted(RuntimeError):
    """A dry run reached a call that is not cached.

    Raised rather than returning a fake answer, because a dry run that
    silently invents responses would produce a report that looks real.
    """


class CachedLLM:
    """Caching, accounting wrapper around any `LLMClient`."""

    def __init__(
        self,
        provider: LLMClient,
        cache: LLMCache | None,
        ledger: CallLedger,
        dry_run: bool = False,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.ledger = ledger
        self.dry_run = dry_run
        #: Calls a dry run would have had to make.
        self.dry_run_misses = 0

    @property
    def model(self) -> str:
        return self.provider.model

    def complete(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        purpose: str = "generic",
        **params: Any,
    ) -> LLMResponse:
        key = cache_key(self.provider.provider, self.provider.model, messages, schema, params)

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.ledger.record(
                    purpose, hit.model or self.provider.model, hit.tokens_in, hit.tokens_out,
                    cached=True,
                )
                return hit

        if self.dry_run:
            self.dry_run_misses += 1
            raise DryRunExhausted(
                f"dry run: {purpose} call is not cached "
                f"(model={self.provider.model}, misses so far={self.dry_run_misses})"
            )

        started = time.perf_counter()
        response = self.provider.complete(messages, schema=schema, **params)
        response.latency_s = time.perf_counter() - started
        response.cached = False

        if self.cache is not None:
            self.cache.put(key, self.provider.provider, self.provider.model, response)
        self.ledger.record(
            purpose,
            response.model or self.provider.model,
            response.tokens_in,
            response.tokens_out,
            cached=False,
            latency_s=response.latency_s,
        )
        return response

    def is_cached(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        **params: Any,
    ) -> bool:
        """Whether this exact call would be free.

        Used by dry runs to count and price the work without making it.
        """
        if self.cache is None:
            return False
        key = cache_key(self.provider.provider, self.provider.model, messages, schema, params)
        return self.cache.get(key) is not None

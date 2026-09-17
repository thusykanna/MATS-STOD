"""Token and cost accounting.

Every call is recorded, cached or not, because the interesting number for the
write-up is what an experiment would cost from cold, while the interesting
number for today's budget is what it actually spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallRecord:
    purpose: str
    model: str
    tokens_in: int
    tokens_out: int
    cached: bool
    latency_s: float = 0.0


@dataclass
class CostLedger:
    """Running totals for one run."""

    prices_usd_per_mtok: dict[str, dict[str, float]] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)

    def record(
        self,
        purpose: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cached: bool,
        latency_s: float = 0.0,
    ) -> None:
        self.calls.append(
            CallRecord(purpose, model, tokens_in, tokens_out, cached, latency_s)
        )

    # -- aggregates -------------------------------------------------------

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Cost of one call, or 0.0 when the model has no price entry.

        A missing price is reported separately by `models_without_prices`.
        Treating it as free here and saying nothing would put "$0.00" in a
        report for a run that really did spend money.
        """
        price = self.prices_usd_per_mtok.get(model)
        if price is None:
            return 0.0
        return (
            tokens_in / 1_000_000 * price.get("input", 0.0)
            + tokens_out / 1_000_000 * price.get("output", 0.0)
        )

    @property
    def models_without_prices(self) -> list[str]:
        """Models that were called but are not in the price table.

        Any cost figure in this run is a lower bound while this is non-empty.
        """
        return sorted(
            {c.model for c in self.calls if c.model not in self.prices_usd_per_mtok}
        )

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_cached(self) -> int:
        return sum(1 for c in self.calls if c.cached)

    @property
    def tokens_in_total(self) -> int:
        return sum(c.tokens_in for c in self.calls)

    @property
    def tokens_out_total(self) -> int:
        return sum(c.tokens_out for c in self.calls)

    @property
    def tokens_in_billed(self) -> int:
        return sum(c.tokens_in for c in self.calls if not c.cached)

    @property
    def tokens_out_billed(self) -> int:
        return sum(c.tokens_out for c in self.calls if not c.cached)

    @property
    def cost_usd_billed(self) -> float:
        """What this run actually spent (cache hits are free)."""
        return sum(
            self._cost(c.model, c.tokens_in, c.tokens_out) for c in self.calls if not c.cached
        )

    @property
    def cost_usd_cold(self) -> float:
        """What this run would cost with an empty cache."""
        return sum(self._cost(c.model, c.tokens_in, c.tokens_out) for c in self.calls)

    @property
    def latency_s_total(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def by_purpose(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.calls:
            slot = out.setdefault(
                c.purpose,
                {"calls": 0, "cached": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd_cold": 0.0},
            )
            slot["calls"] += 1
            slot["cached"] += int(c.cached)
            slot["tokens_in"] += c.tokens_in
            slot["tokens_out"] += c.tokens_out
            slot["cost_usd_cold"] += self._cost(c.model, c.tokens_in, c.tokens_out)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.n_calls,
            "cache_hits": self.n_cached,
            "tokens_in_total": self.tokens_in_total,
            "tokens_out_total": self.tokens_out_total,
            "tokens_in_billed": self.tokens_in_billed,
            "tokens_out_billed": self.tokens_out_billed,
            "cost_usd_billed": round(self.cost_usd_billed, 6),
            "cost_usd_cold": round(self.cost_usd_cold, 6),
            "models_without_prices": self.models_without_prices,
            "cost_is_complete": not self.models_without_prices,
            "latency_s_total": round(self.latency_s_total, 3),
            "by_purpose": self.by_purpose(),
        }

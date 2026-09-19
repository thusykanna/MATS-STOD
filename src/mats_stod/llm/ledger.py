"""Call accounting.

Every call is recorded, cached or not, because the interesting number for the
write-up is how many calls an experiment made and how many were served from
cache, not what it cost.
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
class CallLedger:
    """Running totals for one run."""

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
    def latency_s_total(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def by_purpose(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.calls:
            slot = out.setdefault(
                c.purpose,
                {"calls": 0, "cached": 0, "tokens_in": 0, "tokens_out": 0},
            )
            slot["calls"] += 1
            slot["cached"] += int(c.cached)
            slot["tokens_in"] += c.tokens_in
            slot["tokens_out"] += c.tokens_out
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.n_calls,
            "cache_hits": self.n_cached,
            "tokens_in_total": self.tokens_in_total,
            "tokens_out_total": self.tokens_out_total,
            "tokens_in_billed": self.tokens_in_billed,
            "tokens_out_billed": self.tokens_out_billed,
            "latency_s_total": round(self.latency_s_total, 3),
            "by_purpose": self.by_purpose(),
        }

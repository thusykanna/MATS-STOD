"""Context strategies.

A strategy decides two things: the order segments are translated in, and what
each one is shown alongside its own text. Everything else about a translation
run is held constant, so a difference between conditions is attributable to
context and nothing else. The DAG strategies (D1, D2) land here later and
implement the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import Settings
from ..io.text import estimate_tokens
from ..schemas import Segment, TranslationRecord


@dataclass
class ContextBlock:
    """What a segment is shown before its own text."""

    text: str
    seg_ids: list[str] = field(default_factory=list)
    truncated: bool = False


@dataclass
class TranslationState:
    """What has been produced so far in this document."""

    segments: list[Segment]
    records: dict[str, TranslationRecord] = field(default_factory=dict)

    def by_id(self, seg_id: str) -> Segment:
        for s in self.segments:
            if s.seg_id == seg_id:
                return s
        raise KeyError(seg_id)

    def translated(self, seg_id: str) -> str | None:
        rec = self.records.get(seg_id)
        return rec.target_text if rec else None


class ContextStrategy(ABC):
    """Interface shared by every experiment condition."""

    name: str = "base"
    #: Whether this strategy needs a discourse graph. Baselines do not.
    needs_graph: bool = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def order(self, state: TranslationState) -> list[str]:
        """Segment ids in the order they must be translated."""

    @abstractmethod
    def build_context(self, seg_id: str, state: TranslationState) -> ContextBlock:
        """The context block shown with `seg_id`."""

    # -- shared helpers ---------------------------------------------------

    def _pair_lines(self, seg_ids: list[str], state: TranslationState) -> list[str]:
        """Render earlier segments as source/translation pairs.

        The translation is included when available, because consistency is the
        thing context is supposed to buy: showing only the source tells the
        model what was said, not how it was rendered.
        """
        lines: list[str] = []
        for sid in seg_ids:
            seg = state.by_id(sid)
            lines.append(f"[{seg.order}] source: {seg.text}")
            produced = state.translated(sid)
            if produced:
                lines.append(f"[{seg.order}] translation: {produced}")
        return lines

    def _fit_budget(self, seg_ids: list[str], state: TranslationState) -> ContextBlock:
        """Drop the oldest context until it fits the token budget.

        Oldest first, because the nearest context is the most likely to carry
        the pronoun or term the current segment depends on.
        """
        budget = self.settings.translation.context_token_budget
        cpt = self.settings.translation.chars_per_token_estimate
        kept = list(seg_ids)
        truncated = False
        while kept:
            text = "\n".join(self._pair_lines(kept, state))
            if estimate_tokens(text, cpt) <= budget:
                return ContextBlock(text=text, seg_ids=kept, truncated=truncated)
            kept.pop(0)
            truncated = True
        return ContextBlock(text="", seg_ids=[], truncated=truncated)


class B0FullDocument(ContextStrategy):
    """Whole document in one prompt.

    The document is the unit of translation, so there is no context block: the
    model sees everything at once. When the document exceeds the configured
    token limit it is cut into fixed chunks, and that fallback is recorded,
    because a B0 that silently became chunked is a different condition.
    """

    name = "B0_full_document"

    def order(self, state: TranslationState) -> list[str]:
        return [s.seg_id for s in sorted(state.segments, key=lambda s: s.order)]

    def build_context(self, seg_id: str, state: TranslationState) -> ContextBlock:
        return ContextBlock(text="", seg_ids=[])


class B1Isolated(ContextStrategy):
    """Each segment alone, no context. The floor every other condition beats."""

    name = "B1_isolated"

    def order(self, state: TranslationState) -> list[str]:
        return [s.seg_id for s in sorted(state.segments, key=lambda s: s.order)]

    def build_context(self, seg_id: str, state: TranslationState) -> ContextBlock:
        return ContextBlock(text="", seg_ids=[])


class B2SlidingWindow(ContextStrategy):
    """The previous k segments and their translations.

    This is the baseline that matters: it is cheap, it is what most systems
    do, and the DAG has to beat it to be worth its extra calls.
    """

    name = "B2_sliding_window"

    def order(self, state: TranslationState) -> list[str]:
        return [s.seg_id for s in sorted(state.segments, key=lambda s: s.order)]

    def build_context(self, seg_id: str, state: TranslationState) -> ContextBlock:
        ordered = sorted(state.segments, key=lambda s: s.order)
        position = next(i for i, s in enumerate(ordered) if s.seg_id == seg_id)
        k = self.settings.translation.window_k
        window = [s.seg_id for s in ordered[max(0, position - k) : position]]
        if not window:
            return ContextBlock(text="", seg_ids=[])
        return self._fit_budget(window, state)


_STRATEGIES: dict[str, type[ContextStrategy]] = {
    B0FullDocument.name: B0FullDocument,
    B1Isolated.name: B1Isolated,
    B2SlidingWindow.name: B2SlidingWindow,
}


def build_strategy(name: str, settings: Settings) -> ContextStrategy:
    """Instantiate a context strategy by config name."""
    short = {"B0": B0FullDocument.name, "B1": B1Isolated.name, "B2": B2SlidingWindow.name}
    resolved = short.get(name, name)
    if resolved not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy: {name!r}. Available: {sorted(_STRATEGIES)} "
            "(D1 and D2 arrive with the DAG pipeline)"
        )
    return _STRATEGIES[resolved](settings)


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES)

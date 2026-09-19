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

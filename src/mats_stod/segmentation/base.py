"""Segmenter interface and shared helpers.

Kept as an interface, not a registry: GRAFT is the only segmenter for now, but
a future comparison method implements this ABC rather than being wired in ad
hoc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from ..schemas import Document, Segment


@dataclass
class SegmentationResult:
    """Segments plus everything worth reporting about how they were produced."""

    segments: list[Segment]
    stats: dict[str, Any] = field(default_factory=dict)


class Segmenter(ABC):
    """Turns a parsed `Document` into ordered, non-overlapping segments."""

    name: str = "base"

    @abstractmethod
    def segment(self, document: Document) -> SegmentationResult:
        """Produce segments whose offsets index `document.raw_text`."""


def make_segment(
    document: Document,
    order: int,
    char_start: int,
    char_end: int,
    block_ids: list[str],
    seg_type: str = "paragraph",
    heading_path: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Segment:
    """Build a Segment, slicing its text from the document.

    Text is always sliced rather than passed in, so a segment can never hold
    text that differs from the characters its offsets point at.
    """
    return Segment(
        seg_id=f"{document.doc_id}-s{order:04d}",
        doc_id=document.doc_id,
        order=order,
        text=document.raw_text[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
        block_ids=list(block_ids),
        seg_type=seg_type,  # type: ignore[arg-type]
        heading_path=list(heading_path or []),
        metadata=dict(metadata or {}),
    )


def segment_length_stats(segments: list[Segment]) -> dict[str, Any]:
    """Length statistics reported for every segmentation run."""
    if not segments:
        return {"n_segments": 0}
    lengths = [len(s.text) for s in segments]
    return {
        "n_segments": len(segments),
        "chars_total": sum(lengths),
        "chars_mean": round(mean(lengths), 1),
        "chars_median": round(median(lengths), 1),
        "chars_min": min(lengths),
        "chars_max": max(lengths),
    }

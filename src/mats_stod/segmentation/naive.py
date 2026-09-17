"""Naive blank-line segmentation used by the Phase A baselines.

B1 and B2 are defined in the build plan as operating on naively split
paragraphs, not on the researched segmentations. Keeping that split here, and
separate from `StructuralSegmenter`, means an improvement to structural
segmentation cannot quietly improve the baselines it is measured against.
"""

from __future__ import annotations

from ..parsing.plaintext import _paragraph_spans
from ..schemas import Document, Segment
from .base import SegmentationResult, Segmenter, make_segment, segment_length_stats


class NaiveParagraphSegmenter(Segmenter):
    """One segment per blank-line-delimited paragraph. No LLM, no merging."""

    name = "naive_paragraph"

    def segment(self, document: Document) -> SegmentationResult:
        segments: list[Segment] = []
        for order, (start, end) in enumerate(_paragraph_spans(document.raw_text)):
            segments.append(
                make_segment(
                    document,
                    order=order,
                    char_start=start,
                    char_end=end,
                    block_ids=[],
                    seg_type="paragraph",
                    metadata={"segmenter": self.name},
                )
            )
        stats = segment_length_stats(segments)
        stats.update({"segmenter": self.name, "llm_calls": 0})
        return SegmentationResult(segments=segments, stats=stats)


class WholeDocumentSegmenter(Segmenter):
    """The entire document as one segment, for B0."""

    name = "whole_document"

    def segment(self, document: Document) -> SegmentationResult:
        text = document.raw_text
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        segments = (
            [
                make_segment(
                    document,
                    order=0,
                    char_start=start,
                    char_end=end,
                    block_ids=[b.block_id for b in document.blocks],
                    seg_type="other",
                    metadata={"segmenter": self.name},
                )
            ]
            if end > start
            else []
        )
        stats = segment_length_stats(segments)
        stats.update({"segmenter": self.name, "llm_calls": 0})
        return SegmentationResult(segments=segments, stats=stats)


def chunk_document(document: Document, max_chars: int) -> SegmentationResult:
    """Split a document into fixed chunks at paragraph edges.

    Used only when B0 exceeds its token limit. Chunking at paragraph edges
    rather than at a character count keeps invariant 1 intact and avoids
    cutting a sentence in half.
    """
    spans = _paragraph_spans(document.raw_text)
    segments: list[Segment] = []
    if not spans:
        return SegmentationResult(segments=[], stats={"segmenter": "chunked", "llm_calls": 0})

    chunk_start, chunk_end = spans[0]
    for start, end in spans[1:]:
        if end - chunk_start > max_chars:
            segments.append(
                make_segment(
                    document, len(segments), chunk_start, chunk_end, [], "other",
                    metadata={"segmenter": "chunked"},
                )
            )
            chunk_start = start
        chunk_end = end
    segments.append(
        make_segment(
            document, len(segments), chunk_start, chunk_end, [], "other",
            metadata={"segmenter": "chunked"},
        )
    )
    stats = segment_length_stats(segments)
    stats.update({"segmenter": "chunked", "llm_calls": 0, "max_chars": max_chars})
    return SegmentationResult(segments=segments, stats=stats)

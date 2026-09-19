"""Whole-document and fixed-chunk segmentation used by the B0 translation condition."""

from __future__ import annotations

from ..parsing.plaintext import _paragraph_spans
from ..schemas import Document, Segment
from .base import SegmentationResult, Segmenter, make_segment, segment_length_stats


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

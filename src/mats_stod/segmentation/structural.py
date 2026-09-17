"""Rules-only segmenter.

One segment per layout block, with short blocks merged and long blocks split
at sentence boundaries. This is the zero-cost baseline the GRAFT segmenter has
to beat, and it is also the starting point the GRAFT segmenter refines, so its
behaviour has to be boring and predictable.
"""

from __future__ import annotations

from ..config import SegmentationSettings
from ..schemas import Document, LayoutBlock, Segment
from .base import SegmentationResult, Segmenter, make_segment, segment_length_stats
from .sentences import SentenceSplitter


class StructuralSegmenter(Segmenter):
    name = "structural"

    def __init__(self, settings: SegmentationSettings | None = None) -> None:
        self.settings = settings or SegmentationSettings()
        self.splitter = SentenceSplitter(self.settings.sentences)

    def segment(self, document: Document) -> SegmentationResult:
        spans = self._block_spans(document)
        if self.settings.merge_short_blocks:
            spans = self._merge_short(spans, document)
        if self.settings.split_long_blocks:
            spans = self._split_long(spans, document)

        segments: list[Segment] = []
        for order, (start, end, block_ids, seg_type) in enumerate(spans):
            segments.append(
                make_segment(
                    document,
                    order=order,
                    char_start=start,
                    char_end=end,
                    block_ids=block_ids,
                    seg_type=seg_type,
                    heading_path=[],  # layout parsing deferred: no headings yet
                    metadata={"segmenter": self.name},
                )
            )

        stats = segment_length_stats(segments)
        stats.update(
            {
                "segmenter": self.name,
                "n_blocks": len(document.blocks),
                "llm_calls": 0,
                "layout_detected": bool(document.metadata.get("layout_detected")),
            }
        )
        return SegmentationResult(segments=segments, stats=stats)

    # -- steps ------------------------------------------------------------

    def _block_spans(self, document: Document) -> list[tuple[int, int, list[str], str]]:
        if document.blocks:
            return [(b.char_start, b.char_end, [b.block_id], b.type) for b in document.blocks]
        # A document with no blocks is still segmentable; treat it as one block.
        text = document.raw_text
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        return [(start, end, [], "paragraph")] if end > start else []

    def _merge_short(
        self, spans: list[tuple[int, int, list[str], str]], document: Document
    ) -> list[tuple[int, int, list[str], str]]:
        """Fold blocks below the minimum length into the following block.

        Merging forward rather than backward keeps a short lead-in line
        attached to what it introduces, which is the usual shape in a circular.
        """
        out: list[tuple[int, int, list[str], str]] = []
        pending: tuple[int, int, list[str], str] | None = None
        for span in spans:
            start, end, ids, seg_type = span
            if pending is not None:
                start = pending[0]
                ids = pending[2] + ids
                seg_type = pending[3] if pending[3] != "paragraph" else seg_type
                pending = None
            body = document.raw_text[start:end]
            if len(body.strip()) < self.settings.min_segment_chars:
                pending = (start, end, ids, seg_type)
                continue
            out.append((start, end, ids, seg_type))
        if pending is not None:
            if out:
                last = out[-1]
                out[-1] = (last[0], pending[1], last[2] + pending[2], last[3])
            else:
                out.append(pending)
        return out

    def _split_long(
        self, spans: list[tuple[int, int, list[str], str]], document: Document
    ) -> list[tuple[int, int, list[str], str]]:
        """Split over-long blocks at sentence boundaries, never mid-sentence."""
        out: list[tuple[int, int, list[str], str]] = []
        for start, end, ids, seg_type in spans:
            body = document.raw_text[start:end]
            if len(body) <= self.settings.max_segment_chars:
                out.append((start, end, ids, seg_type))
                continue
            sentences = self.splitter.split(body, offset=start)
            if len(sentences) <= 1:
                # Nothing to split on: an over-long single sentence stays whole
                # rather than being cut at an arbitrary character.
                out.append((start, end, ids, seg_type))
                continue
            chunk_start = sentences[0].char_start
            chunk_end = sentences[0].char_end
            for sentence in sentences[1:]:
                if sentence.char_end - chunk_start > self.settings.max_segment_chars:
                    out.append((chunk_start, chunk_end, ids, seg_type))
                    chunk_start = sentence.char_start
                chunk_end = sentence.char_end
            out.append((chunk_start, chunk_end, ids, seg_type))
        return out


def block_of(document: Document, char_start: int) -> LayoutBlock | None:
    for b in document.blocks:
        if b.char_start <= char_start < b.char_end:
            return b
    return None

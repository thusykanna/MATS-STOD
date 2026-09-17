"""GRAFT's discourse agent.

Faithful to Dutta et al. (2025): a discourse starts at one sentence and grows
greedily, asking the model once per candidate next sentence whether it belongs
to the discourse being built. One "no" closes the discourse and starts the
next. Cost is one call per sentence boundary considered, which is linear in
the document.

Two departures from the reference implementation, both forced by this
project's invariants rather than by taste:

1. The model never sees or returns text to keep. It answers yes or no, and
   the segment is assembled from the sentence spans that were already computed
   from the source string. A model cannot therefore rewrite the document, and
   invariant 1 holds without trusting it.
2. Growth never crosses a layout block whose type is atomic (headings, list
   items, table cells). GRAFT works on flat prose; official documents are not
   flat, and gluing a heading onto the paragraph below it destroys the only
   structure the plain-text parser preserves.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..llm.base import Message, ParseError, parse_yes_no
from ..llm.client import CachedLLM
from ..prompts.registry import render
from ..schemas import Document, Segment
from .base import SegmentationResult, Segmenter, make_segment, segment_length_stats
from .sentences import SentenceSpan, SentenceSplitter


@dataclass
class _Unit:
    """A sentence plus the block it came from."""

    span: SentenceSpan
    block_id: str
    block_type: str


class GraftDiscourseSegmenter(Segmenter):
    name = "graft"

    def __init__(self, settings: Settings, llm: CachedLLM) -> None:
        self.settings = settings
        self.seg_settings = settings.segmentation
        self.llm = llm
        self.splitter = SentenceSplitter(self.seg_settings.sentences)

    def segment(self, document: Document) -> SegmentationResult:
        units = self._sentence_units(document)
        if not units:
            return SegmentationResult(segments=[], stats={"segmenter": self.name, "llm_calls": 0})

        atomic = set(self.seg_settings.atomic_types)
        groups = _group_units(units, atomic)

        segments: list[Segment] = []
        calls = 0
        cache_hits = 0
        unparsed = 0
        forced_by_cap = 0
        yes_count = 0

        for group in groups:
            i = 0
            while i < len(group):
                current = [group[i]]
                j = i + 1
                while j < len(group):
                    candidate = group[j]
                    start = current[0].span.char_start
                    end_if_added = candidate.span.char_end
                    if end_if_added - start > self.seg_settings.max_discourse_chars:
                        # Hard cap: one runaway chain of "yes" answers must not
                        # swallow the document.
                        forced_by_cap += 1
                        break

                    discourse_text = document.raw_text[
                        current[0].span.char_start : current[-1].span.char_end
                    ]
                    next_text = candidate.span.text_of(document.raw_text)
                    decision, was_cached, ok = self._ask(discourse_text, next_text)
                    calls += 1
                    cache_hits += int(was_cached)
                    if not ok:
                        unparsed += 1
                    if not decision:
                        break
                    yes_count += 1
                    current.append(candidate)
                    j += 1

                segments.append(
                    self._make(document, len(segments), current)
                )
                i = j if j > i else i + 1

        stats = segment_length_stats(segments)
        stats.update(
            {
                "segmenter": self.name,
                "n_sentence_units": len(units),
                "n_groups": len(groups),
                "llm_calls": calls,
                "llm_cache_hits": cache_hits,
                "decisions_yes": yes_count,
                "decisions_unparsed": unparsed,
                "boundaries_forced_by_cap": forced_by_cap,
                "prompt_version": self.seg_settings.prompt_version,
                "layout_detected": bool(document.metadata.get("layout_detected")),
            }
        )
        return SegmentationResult(segments=segments, stats=stats)

    # -- helpers ----------------------------------------------------------

    def _ask(self, discourse: str, next_sentence: str) -> tuple[bool, bool, bool]:
        """Ask whether `next_sentence` continues `discourse`.

        Returns (decision, cached, parsed_ok). An unparseable answer counts as
        a boundary: the conservative reading, and cheaper to defend than a
        retry that doubles the cost of the whole segmentation pass.
        """
        prompt = render(
            self.seg_settings.prompt_version,
            source_lang=self.settings.langs.source_name,
            target_lang=self.settings.langs.target_name,
            discourse=discourse,
            next_sentence=next_sentence,
        )
        response = self.llm.complete(
            [Message("user", prompt)],
            purpose="segmentation",
            # GRAFT uses a single greedy token; anything longer is wasted spend.
            max_output_tokens=4,
            temperature=0.0,
        )
        try:
            return parse_yes_no(response.text), response.cached, True
        except ParseError:
            return False, response.cached, False

    def _make(self, document: Document, order: int, units: list[_Unit]) -> Segment:
        start = units[0].span.char_start
        end = units[-1].span.char_end
        block_ids: list[str] = []
        for u in units:
            if u.block_id and u.block_id not in block_ids:
                block_ids.append(u.block_id)
        return make_segment(
            document,
            order=order,
            char_start=start,
            char_end=end,
            block_ids=block_ids,
            seg_type=units[0].block_type,
            heading_path=[],  # layout parsing deferred
            metadata={"segmenter": self.name, "n_sentences": len(units)},
        )

    def _sentence_units(self, document: Document) -> list[_Unit]:
        """Sentence spans for the whole document, tagged with their block."""
        units: list[_Unit] = []
        blocks = document.blocks or []
        if not blocks:
            for span in self.splitter.split(document.raw_text):
                units.append(_Unit(span, "", "paragraph"))
            return units

        for block in blocks:
            body = block.text_of(document.raw_text)
            for span in self.splitter.split(body, offset=block.char_start):
                units.append(_Unit(span, block.block_id, block.type))
        return units


def _group_units(units: list[_Unit], atomic_types: set[str]) -> list[list[_Unit]]:
    """Split units into runs that a discourse may grow within.

    A block whose type is atomic forms its own group, so a heading or a list
    item is never merged into neighbouring prose.
    """
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if unit.block_type in atomic_types:
            if current:
                groups.append(current)
                current = []
            groups.append([unit])
            continue
        current.append(unit)
    if current:
        groups.append(current)
    return groups

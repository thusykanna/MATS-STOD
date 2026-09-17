"""Phase A baseline pipeline.

Plain functions, no graph framework: B0, B1 and B2 are a loop over segments
with no branching and no state beyond the records produced so far. Wrapping
that in a state machine would add a dependency and explain nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..io.parallel import DocPair
from ..io.runs import RunDir
from ..io.text import estimate_tokens
from ..llm.client import CachedLLM
from ..parsing.plaintext import PlainTextParser
from ..schemas import Document, Segment, TranslationRecord
from ..segmentation.base import SegmentationResult
from ..segmentation.naive import (
    NaiveParagraphSegmenter,
    WholeDocumentSegmenter,
    chunk_document,
)
from ..translation.context import (
    B0FullDocument,
    ContextStrategy,
    TranslationState,
    build_strategy,
)
from ..translation.translator import Translator


@dataclass
class DocumentTranslation:
    """Everything one document produced under one condition."""

    doc_id: str
    direction: str
    strategy: str
    segments: list[Segment]
    records: list[TranslationRecord]
    output_text: str
    reference_text: str
    source_text: str
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def parse_document(pair: DocPair) -> Document:
    """Parse a loaded pair into a Document in the source language."""
    return PlainTextParser().parse(
        pair.source_text,
        doc_id=pair.doc_id,
        source_lang=pair.source_lang,
        direction=pair.direction,
    )


def segment_for_strategy(
    document: Document, strategy: ContextStrategy, settings: Settings
) -> tuple[SegmentationResult, list[str]]:
    """Choose the segmentation a baseline condition requires.

    B0 is one segment unless the document is too large, in which case it
    becomes fixed chunks and the fallback is recorded as a note, because a
    chunked B0 is no longer the condition it claims to be.
    """
    notes: list[str] = []
    if isinstance(strategy, B0FullDocument):
        est = estimate_tokens(
            document.raw_text, settings.translation.chars_per_token_estimate
        )
        if est > settings.translation.full_doc_token_limit:
            max_chars = int(
                settings.translation.chunk_token_size
                * settings.translation.chars_per_token_estimate
            )
            notes.append(
                f"B0 fallback: document estimated at {est} tokens, above the limit of "
                f"{settings.translation.full_doc_token_limit}; translated in fixed chunks "
                f"of at most {max_chars} characters."
            )
            return chunk_document(document, max_chars), notes
        return WholeDocumentSegmenter().segment(document), notes
    return NaiveParagraphSegmenter().segment(document), notes


def translate_document(
    pair: DocPair,
    settings: Settings,
    llm: CachedLLM,
    strategy: ContextStrategy | None = None,
) -> DocumentTranslation:
    """Translate one document under one context strategy."""
    strat = strategy or build_strategy(settings.translation.strategy, settings)
    document = parse_document(pair)
    seg_result, notes = segment_for_strategy(document, strat, settings)
    state = TranslationState(segments=seg_result.segments)
    translator = Translator(settings, llm)

    for seg_id in strat.order(state):
        segment = state.by_id(seg_id)
        context = strat.build_context(seg_id, state)
        record = translator.translate_segment(segment, context, strat.name)
        state.records[seg_id] = record

    ordered = sorted(state.segments, key=lambda s: s.order)
    output = "\n\n".join(state.records[s.seg_id].target_text for s in ordered)

    stats = dict(seg_result.stats)
    stats["n_flagged_segments"] = sum(1 for r in state.records.values() if r.flags)

    return DocumentTranslation(
        doc_id=pair.doc_id,
        direction=pair.direction,
        strategy=strat.name,
        segments=ordered,
        records=[state.records[s.seg_id] for s in ordered],
        output_text=output,
        reference_text=pair.reference_text,
        source_text=pair.source_text,
        notes=notes,
        stats=stats,
    )


def write_document_artifacts(run: RunDir, result: DocumentTranslation) -> None:
    """Persist one document's artifacts so any stage can be re-run alone."""
    docs_dir = run.subdir("documents")
    base = docs_dir / result.doc_id
    base.mkdir(parents=True, exist_ok=True)
    run.write_json(f"documents/{result.doc_id}/segments.json", result.segments)
    run.write_json(f"documents/{result.doc_id}/records.json", result.records)
    run.write_text(f"documents/{result.doc_id}/translation.txt", result.output_text + "\n")
    run.write_json(
        f"documents/{result.doc_id}/stats.json",
        {"stats": result.stats, "notes": result.notes, "direction": result.direction},
    )

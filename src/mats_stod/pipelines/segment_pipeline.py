"""Segmentation pipeline.

Runs a segmenter over a set of documents, enforces the invariants on every
one, scores against gold when it exists, and writes the artifacts. Kept apart
from translation so a segmentation experiment costs only segmentation calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..evaluation.annotate import gold_stem
from ..evaluation.segmentation_score import (
    SegmentationScore,
    build_report_section,
    load_gold_boundaries,
    score_segmentation,
)
from ..io.parallel import DocPair
from ..io.runs import RunDir
from ..llm.client import CachedLLM
from ..parsing.plaintext import PlainTextParser
from ..schemas import DiscourseGraph, Document, Segment
from ..segmentation.base import build_segmenter


@dataclass
class DocumentSegmentation:
    doc_id: str
    direction: str
    document: Document
    segments: list[Segment]
    stats: dict[str, Any] = field(default_factory=dict)


def segment_document(
    pair: DocPair, settings: Settings, llm: CachedLLM | None = None
) -> DocumentSegmentation:
    """Segment one document and check the invariants before returning it."""
    document = PlainTextParser().parse(
        pair.source_text,
        doc_id=pair.doc_id,
        source_lang=pair.source_lang,
        direction=pair.direction,
    )
    segmenter = build_segmenter(settings.segmentation.segmenter, settings, llm)
    result = segmenter.segment(document)

    # Invariants are enforced here, on every document, rather than in tests
    # only: a segmentation that breaks them must not reach translation.
    graph = DiscourseGraph(doc_id=document.doc_id, segments=result.segments)
    graph.validate_graph(raw_text=document.raw_text, allowed_edge_types=settings.graph.edge_types)

    return DocumentSegmentation(
        doc_id=pair.doc_id,
        direction=pair.direction,
        document=document,
        segments=result.segments,
        stats=result.stats,
    )


def score_against_gold(
    seg: DocumentSegmentation, settings: Settings
) -> SegmentationScore | None:
    """Score one document if gold exists for it in this source language.

    Gold is language-specific: the Sinhala and Tamil sides of a pair are
    different documents with different segmentations, so a file that does not
    name the source language is not used.
    """
    stem = gold_stem(seg.doc_id, seg.document.source_lang)
    gold_path = Path(settings.paths.gold_segmentation) / f"{stem}.json"
    if not gold_path.exists():
        return None
    gold = load_gold_boundaries(gold_path)
    return score_segmentation(
        doc_id=seg.doc_id,
        predicted_segments=seg.segments,
        gold_boundaries=gold,
        doc_length=len(seg.document.raw_text),
        tolerance=settings.segmentation.boundary_tolerance_chars,
    )


def run_segmentation(
    pairs: list[DocPair],
    settings: Settings,
    llm: CachedLLM | None,
    run: RunDir,
) -> dict[str, Any]:
    """Segment every document, score where possible, write artifacts."""
    results: list[DocumentSegmentation] = []
    scores: list[SegmentationScore] = []

    for pair in pairs:
        seg = segment_document(pair, settings, llm)
        results.append(seg)
        run.write_json(f"documents/{seg.doc_id}/segments.json", seg.segments)
        run.write_json(f"documents/{seg.doc_id}/segmentation_stats.json", seg.stats)
        run.log(
            "segmented",
            doc_id=seg.doc_id,
            direction=seg.direction,
            n_segments=len(seg.segments),
            llm_calls=seg.stats.get("llm_calls", 0),
        )
        score = score_against_gold(seg, settings)
        if score is not None:
            scores.append(score)

    aggregate_stats = _aggregate_stats(results)
    out: dict[str, Any] = {"segmentation_stats": aggregate_stats}
    if scores:
        out["segmentation_scores"] = build_report_section(scores)
    else:
        out["notes"] = [
            "No gold segmentation files found, so boundary F1, Pk and WindowDiff "
            "were not computed. Create gold files with `mats-stod annotate-template` "
            "and `mats-stod annotate-import`."
        ]
    return out


def _aggregate_stats(results: list[DocumentSegmentation]) -> dict[str, Any]:
    if not results:
        return {"n_documents": 0}
    n_segments = sum(len(r.segments) for r in results)
    lengths = [len(s.text) for r in results for s in r.segments]
    return {
        "n_documents": len(results),
        "segmenter": results[0].stats.get("segmenter"),
        "n_segments_total": n_segments,
        "segments_per_document": round(n_segments / len(results), 2),
        "chars_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "chars_min": min(lengths) if lengths else 0,
        "chars_max": max(lengths) if lengths else 0,
        "llm_calls_total": sum(r.stats.get("llm_calls", 0) for r in results),
        "llm_cache_hits_total": sum(r.stats.get("llm_cache_hits", 0) for r in results),
        "decisions_unparsed_total": sum(r.stats.get("decisions_unparsed", 0) for r in results),
        "boundaries_forced_by_cap_total": sum(
            r.stats.get("boundaries_forced_by_cap", 0) for r in results
        ),
        "layout_detected": bool(results[0].stats.get("layout_detected")),
    }

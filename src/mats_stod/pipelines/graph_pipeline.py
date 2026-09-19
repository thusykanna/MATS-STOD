"""Graph-building pipeline.

Segments a document, runs pass 1 (deterministic structural edges) and pass 2
(the configured inferrer), assembles the result and writes the artifacts.
Kept apart from translation so that building a graph costs only edge calls,
and so a graph can be rebuilt from a cache without retranslating anything.

Structural edges are merged first, so that when a deterministic edge and an
inferred one describe the same pair the deterministic one is the copy that
survives deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..evaluation.annotate import gold_stem
from ..evaluation.edge_score import (
    EdgeScore,
    build_report_section,
    load_gold_edges,
    score_edges,
)
from ..graph.assemble import assemble_graph
from ..graph.export import write_graphml, write_json, write_mermaid
from ..graph.graft_edges import GraftPairwiseEdgeInferrer
from ..graph.stats import aggregate_graph_stats, graph_stats
from ..graph.structural_edges import infer_structural_edges
from ..io.parallel import DocPair
from ..io.runs import RunDir
from ..llm.client import CachedLLM
from ..schemas import DiscourseGraph
from .segment_pipeline import segment_document


@dataclass
class DocumentGraph:
    doc_id: str
    direction: str
    source_lang: str
    graph: DiscourseGraph
    stats: dict[str, Any] = field(default_factory=dict)


def build_document_graph(pair: DocPair, settings: Settings, llm: CachedLLM) -> DocumentGraph:
    """Segment one document and build its discourse graph."""
    seg = segment_document(pair, settings, llm)
    inferrer = GraftPairwiseEdgeInferrer(settings, llm)

    structural = infer_structural_edges(seg.document, seg.segments)
    inferred = inferrer.infer(seg.segments)

    graph, assembly = assemble_graph(
        doc_id=seg.doc_id,
        segments=seg.segments,
        proposed=[*structural, *inferred.edges],
        settings=settings,
        raw_text=seg.document.raw_text,
    )

    stats = graph_stats(graph)
    stats.update(
        {
            "edge_inferrer": inferrer.name,
            "segmenter": seg.stats.get("segmenter"),
            "n_structural_proposed": len(structural),
            "n_inferred_proposed": len(inferred.edges),
            "assembly": assembly.as_dict(),
        }
    )
    stats.update({f"inferrer_{k}": v for k, v in inferred.stats.items() if k != "inferrer"})

    return DocumentGraph(
        doc_id=seg.doc_id,
        direction=seg.direction,
        source_lang=seg.document.source_lang,
        graph=graph,
        stats=stats,
    )


def score_against_gold(doc: DocumentGraph, settings: Settings) -> EdgeScore | None:
    """Score one graph if gold edges exist for it in this source language.

    Gold is language-specific for the same reason segmentation gold is: the
    two sides of a pair are different documents, annotated separately.
    """
    stem = gold_stem(doc.doc_id, doc.source_lang)
    gold_path = Path(settings.paths.gold_edges) / f"{stem}.json"
    if not gold_path.exists():
        return None
    return score_edges(doc.doc_id, doc.graph, load_gold_edges(gold_path))


def run_graph_build(
    pairs: list[DocPair],
    settings: Settings,
    llm: CachedLLM,
    run: RunDir,
) -> dict[str, Any]:
    """Build a graph for every document, score where possible, write artifacts."""
    results: list[DocumentGraph] = []
    scores: list[EdgeScore] = []

    for pair in pairs:
        doc = build_document_graph(pair, settings, llm)
        results.append(doc)

        base = run.subdir("documents") / doc.doc_id
        write_json(doc.graph, base / "graph.json")
        write_mermaid(doc.graph, base / "graph.mmd")
        write_graphml(doc.graph, base / "graph.graphml")
        run.write_json(f"documents/{doc.doc_id}/graph_stats.json", doc.stats)
        run.log(
            "graph_built",
            doc_id=doc.doc_id,
            direction=doc.direction,
            n_edges=doc.stats["n_edges"],
            adjacent_only_share=doc.stats["adjacent_only_share"],
            llm_calls=doc.stats.get("inferrer_llm_calls", 0),
        )

        score = score_against_gold(doc, settings)
        if score is not None:
            scores.append(score)

    out: dict[str, Any] = {"graph_stats": _aggregate(results)}
    if scores:
        out["edge_scores"] = build_report_section(scores)
    else:
        out["notes"] = [
            "No gold edge files found, so edge precision, recall and F1 were not "
            "computed. Create gold files with `mats-stod annotate-template` and "
            "`mats-stod annotate-import`."
        ]
    return out


def _aggregate(results: list[DocumentGraph]) -> dict[str, Any]:
    if not results:
        return {"n_documents": 0}
    agg = aggregate_graph_stats([r.stats for r in results])
    agg["edge_inferrer"] = results[0].stats.get("edge_inferrer")
    agg["segmenter"] = results[0].stats.get("segmenter")
    agg["llm_calls_total"] = sum(r.stats.get("inferrer_llm_calls", 0) for r in results)
    return agg

"""Edge scoring against a hand-annotated gold graph.

Two scores are reported for every run. The typed score requires the inferrer
to name the relation; the untyped score only asks whether the two segments are
connected at all.

The distinction is not decoration. GRAFT's edge agent answers one yes/no
question and therefore cannot name a relation — every edge it emits carries
`graft_dependency` (D6). Scoring it against typed gold would report a
precision of zero for a method that may be finding the right pairs, so the
untyped score is the one to read for GRAFT, and the typed score is what a
future typed inferrer would be held to. Reporting both keeps that honest in
either direction.

Gold files are written by `mats-stod annotate-import` and address segments by
index, which is `Segment.order`; predictions address them by `seg_id`. Both
are converted to order pairs before comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ..schemas import DiscourseGraph


@dataclass
class EdgeScore:
    doc_id: str
    precision: float
    recall: float
    f1: float
    precision_untyped: float
    recall_untyped: float
    f1_untyped: float
    n_gold: int
    n_predicted: int


def load_gold_edges(path: str | Path) -> list[dict[str, Any]]:
    """Read a gold edge file written by `annotate-import`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    return list(data.get("edges", []))


def gold_pairs(
    gold: list[dict[str, Any]],
) -> tuple[set[tuple[int, int, str]], set[tuple[int, int]]]:
    """Gold edges as typed and untyped order pairs."""
    typed: set[tuple[int, int, str]] = set()
    untyped: set[tuple[int, int]] = set()
    for e in gold:
        src = int(e["src_segment_index"])
        dst = int(e["dst_segment_index"])
        typed.add((src, dst, str(e.get("type", ""))))
        untyped.add((src, dst))
    return typed, untyped


def predicted_pairs(
    graph: DiscourseGraph,
) -> tuple[set[tuple[int, int, str]], set[tuple[int, int]]]:
    """Predicted edges as typed and untyped order pairs."""
    order_of = {s.seg_id: s.order for s in graph.segments}
    typed: set[tuple[int, int, str]] = set()
    untyped: set[tuple[int, int]] = set()
    for e in graph.edges:
        if e.src not in order_of or e.dst not in order_of:
            continue
        typed.add((order_of[e.src], order_of[e.dst], e.type))
        untyped.add((order_of[e.src], order_of[e.dst]))
    return typed, untyped


def prf(gold: set, predicted: set) -> tuple[float, float, float]:
    """Precision, recall and F1 over two sets.

    An empty gold set and an empty prediction agree perfectly; predicting
    edges where gold has none is precision zero, not a division error.
    """
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    hits = len(gold & predicted)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def score_edges(doc_id: str, graph: DiscourseGraph, gold: list[dict[str, Any]]) -> EdgeScore:
    """Score one document's graph against its gold edges."""
    g_typed, g_untyped = gold_pairs(gold)
    p_typed, p_untyped = predicted_pairs(graph)
    precision, recall, f1 = prf(g_typed, p_typed)
    up, ur, uf1 = prf(g_untyped, p_untyped)
    return EdgeScore(
        doc_id=doc_id,
        precision=precision,
        recall=recall,
        f1=f1,
        precision_untyped=up,
        recall_untyped=ur,
        f1_untyped=uf1,
        n_gold=len(g_untyped),
        n_predicted=len(p_untyped),
    )


def aggregate(scores: list[EdgeScore]) -> dict[str, Any]:
    """Unweighted means over documents, matching the segmentation scorer."""
    if not scores:
        return {"n_documents": 0}
    return {
        "n_documents": len(scores),
        "precision": round(mean(s.precision for s in scores), 4),
        "recall": round(mean(s.recall for s in scores), 4),
        "f1": round(mean(s.f1 for s in scores), 4),
        "precision_untyped": round(mean(s.precision_untyped for s in scores), 4),
        "recall_untyped": round(mean(s.recall_untyped for s in scores), 4),
        "f1_untyped": round(mean(s.f1_untyped for s in scores), 4),
        "n_gold_total": sum(s.n_gold for s in scores),
        "n_predicted_total": sum(s.n_predicted for s in scores),
    }


def build_report_section(scores: list[EdgeScore]) -> dict[str, Any]:
    return {"documents": [asdict(s) for s in scores], "corpus": aggregate(scores)}

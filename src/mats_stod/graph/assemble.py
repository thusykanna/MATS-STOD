"""Turn proposed edges into a validated DiscourseGraph.

Inferrers propose; this module decides. Everything that must hold however the
edges were produced lives here exactly once, so a new inferrer cannot break an
invariant by forgetting a check: endpoints must exist, an edge must point
forward in reading order, duplicates collapse, and no segment carries more
parents than the context budget can afford.

Nothing is discarded quietly. Every edge removed is counted under a name that
says why, because "the graph has 42 edges" and "the graph has 42 edges and
threw away 130" describe different experiments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..schemas import DiscourseGraph, Edge, ForwardRef, Segment

#: How the parent cap chooses which parents to keep. GRAFT's edges carry no
#: confidence, so there is no score to rank by; nearest-by-reading-order is the
#: rule recorded in DECISIONS.md. It is named here, and written into graph
#: metadata, so an ablation can change it and say so.
PARENT_CAP_RULE = "nearest_by_order"


@dataclass
class AssemblyStats:
    n_proposed: int = 0
    n_kept: int = 0
    dropped_unknown_endpoint: int = 0
    dropped_self_loop: int = 0
    dropped_duplicate: int = 0
    dropped_parent_cap: int = 0
    dropped_transitive: int = 0
    n_forward_refs: int = 0
    parent_cap_rule: str = PARENT_CAP_RULE
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "n_proposed": self.n_proposed,
            "n_kept": self.n_kept,
            "dropped_unknown_endpoint": self.dropped_unknown_endpoint,
            "dropped_self_loop": self.dropped_self_loop,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_parent_cap": self.dropped_parent_cap,
            "dropped_transitive": self.dropped_transitive,
            "n_forward_refs": self.n_forward_refs,
            "parent_cap_rule": self.parent_cap_rule,
        }
        out.update(self.extra)
        return out


def assemble_graph(
    doc_id: str,
    segments: list[Segment],
    proposed: list[Edge],
    settings,  # noqa: ANN001 - avoids a config import cycle
    raw_text: str | None = None,
) -> tuple[DiscourseGraph, AssemblyStats]:
    """Merge proposed edges into a validated graph.

    `raw_text` is passed through to validation so invariant 1 is checked
    against the real characters whenever the document is at hand.
    """
    stats = AssemblyStats(n_proposed=len(proposed))
    order_of = {s.seg_id: s.order for s in segments}

    edges, forward_refs = _split_by_direction(proposed, order_of, stats)
    edges = _dedupe(edges, stats)
    edges = _cap_parents(edges, order_of, settings.graph.max_parents, stats)
    if settings.graph.transitive_reduction:
        edges = _transitive_reduction(edges, stats)

    stats.n_kept = len(edges)
    stats.n_forward_refs = len(forward_refs)

    graph = DiscourseGraph(
        doc_id=doc_id,
        segments=sorted(segments, key=lambda s: s.order),
        edges=edges,
        forward_refs=forward_refs,
        metadata={"assembly": stats.as_dict()},
    )
    graph.validate_graph(raw_text=raw_text, allowed_edge_types=settings.graph.edge_types)
    return graph, stats


def _split_by_direction(
    proposed: list[Edge], order_of: dict[str, int], stats: AssemblyStats
) -> tuple[list[Edge], list[ForwardRef]]:
    """Separate backward-pointing proposals from usable edges.

    An edge whose source comes later in reading order is a forward reference:
    the segment depends on something not yet translated. Admitting it would
    allow a cycle, so it is recorded and set aside instead. How often that
    happens is a property of the corpus worth reporting.
    """
    edges: list[Edge] = []
    refs: list[ForwardRef] = []
    for e in proposed:
        if e.src not in order_of or e.dst not in order_of:
            stats.dropped_unknown_endpoint += 1
            continue
        if e.src == e.dst:
            stats.dropped_self_loop += 1
            continue
        if order_of[e.src] < order_of[e.dst]:
            edges.append(e)
        else:
            refs.append(
                ForwardRef(
                    src=e.src,
                    dst=e.dst,
                    type=e.type,
                    evidence=e.evidence,
                    origin=e.origin,
                    confidence=e.confidence,
                )
            )
    return edges, refs


def _dedupe(edges: list[Edge], stats: AssemblyStats) -> list[Edge]:
    """Collapse edges sharing (src, dst, type), keeping the first seen.

    Order of proposal therefore decides which copy survives. Structural edges
    are merged before LLM ones so a deterministic edge wins over a sampled
    duplicate of itself.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[Edge] = []
    for e in edges:
        if e.key() in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(e.key())
        out.append(e)
    return out


def _cap_parents(
    edges: list[Edge], order_of: dict[str, int], max_parents: int, stats: AssemblyStats
) -> list[Edge]:
    """Keep at most `max_parents` parents per segment, nearest first.

    The cap exists because context is budgeted: a segment with eleven parents
    cannot fit them all into the prompt, and truncating inside the prompt
    builder would make the graph and the context silently disagree.
    """
    if max_parents is None or max_parents <= 0:
        return edges

    by_dst: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        by_dst[e.dst].append(e)

    keep: set[int] = set()
    for group in by_dst.values():
        # Nearest parent first: highest source order is closest to `dst`.
        ranked = sorted(group, key=lambda e: (-order_of[e.src], e.src, e.type))
        kept_parents: list[str] = []
        for e in ranked:
            if e.src in kept_parents:
                keep.add(id(e))
                continue
            if len(kept_parents) >= max_parents:
                stats.dropped_parent_cap += 1
                continue
            kept_parents.append(e.src)
            keep.add(id(e))
    return [e for e in edges if id(e) in keep]


def _transitive_reduction(edges: list[Edge], stats: AssemblyStats) -> list[Edge]:
    """Remove edges implied by a longer path through the graph.

    Off by default. It makes the picture readable, but it deletes exactly the
    short-range edges a sliding window would already have supplied, which
    changes what D1 is being credited for. Turn it on as an ablation, not as
    tidying.
    """
    import networkx as nx

    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e.src, e.dst)
    reduced = nx.transitive_reduction(g)
    surviving = set(reduced.edges())
    out = [e for e in edges if (e.src, e.dst) in surviving]
    stats.dropped_transitive = len(edges) - len(out)
    return out

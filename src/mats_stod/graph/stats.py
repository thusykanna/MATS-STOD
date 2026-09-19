"""What a built graph actually looks like.

One number here decides whether the DAG condition is worth running at all:
the share of edges that only connect neighbours. An edge from segment 5 to
segment 6 supplies context a sliding window already had, so a graph made
entirely of those cannot beat B2 no matter how accurate it is. The share is
reported on every build so that result arrives before the translation bill,
not after it.

`segments_with_nonadjacent_parent` is the same idea at segment level: it names
the subset on which D1 must win if the graph is doing anything. M4's
diagnostics split on exactly this.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from ..schemas import DiscourseGraph


def graph_stats(graph: DiscourseGraph) -> dict[str, Any]:
    """Structural statistics for one document's graph."""
    segments = sorted(graph.segments, key=lambda s: s.order)
    order_of = {s.seg_id: s.order for s in segments}
    n_segments = len(segments)
    edges = graph.edges

    spans = [order_of[e.dst] - order_of[e.src] for e in edges]
    adjacent = sum(1 for d in spans if d == 1)

    parents_of = {s.seg_id: graph.parents(s.seg_id) for s in segments}
    in_degrees = [len(p) for p in parents_of.values()]
    nonadjacent_parent = [
        sid
        for sid, ps in parents_of.items()
        if any(order_of[sid] - order_of[p] > 1 for p in ps)
    ]
    isolated = [
        s.seg_id for s in segments if not parents_of[s.seg_id] and not graph.children(s.seg_id)
    ]

    return {
        "n_segments": n_segments,
        "n_edges": len(edges),
        "edges_by_type": dict(Counter(e.type for e in edges)),
        "edges_by_origin": dict(Counter(e.origin for e in edges)),
        "n_forward_refs": len(graph.forward_refs),
        # The decisive number: 1.0 means the graph says nothing a window did
        # not already say.
        "adjacent_only_share": round(adjacent / len(edges), 4) if edges else 0.0,
        "n_adjacent_edges": adjacent,
        "n_nonadjacent_edges": len(edges) - adjacent,
        "edge_span_mean": round(mean(spans), 2) if spans else 0.0,
        "edge_span_max": max(spans) if spans else 0,
        "in_degree_mean": round(mean(in_degrees), 2) if in_degrees else 0.0,
        "in_degree_max": max(in_degrees) if in_degrees else 0,
        # The M4 subset: segments a sliding window cannot serve.
        "segments_with_nonadjacent_parent": len(nonadjacent_parent),
        "share_with_nonadjacent_parent": (
            round(len(nonadjacent_parent) / n_segments, 4) if n_segments else 0.0
        ),
        "n_isolated_segments": len(isolated),
        "depth": _longest_path(graph),
    }


def _longest_path(graph: DiscourseGraph) -> int:
    """Longest dependency chain, in edges.

    Depth bounds how much context D1 can ever accumulate, so a graph that is
    wide but flat and one that is deep behave differently under the same
    `dag_context_depth`.
    """
    if not graph.edges:
        return 0
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(s.seg_id for s in graph.segments)
    g.add_edges_from((e.src, e.dst) for e in graph.edges)
    return nx.dag_longest_path_length(g)


def aggregate_graph_stats(per_doc: list[dict[str, Any]]) -> dict[str, Any]:
    """Corpus totals over per-document stats."""
    if not per_doc:
        return {"n_documents": 0}
    edges = sum(d["n_edges"] for d in per_doc)
    adjacent = sum(d["n_adjacent_edges"] for d in per_doc)
    segments = sum(d["n_segments"] for d in per_doc)
    with_nonadj = sum(d["segments_with_nonadjacent_parent"] for d in per_doc)
    return {
        "n_documents": len(per_doc),
        "n_segments_total": segments,
        "n_edges_total": edges,
        "adjacent_only_share": round(adjacent / edges, 4) if edges else 0.0,
        "share_with_nonadjacent_parent": round(with_nonadj / segments, 4) if segments else 0.0,
        "n_forward_refs_total": sum(d["n_forward_refs"] for d in per_doc),
        "depth_max": max(d["depth"] for d in per_doc),
    }

"""Write a graph out in three forms, for three different readers.

JSON is the artifact every later stage reads. GraphML is for a graph tool.
Mermaid is for a person: a picture in the report is the fastest way to see
that an inferrer has produced something absurd, and it renders in Markdown
without a plotting dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import DiscourseGraph

#: Characters of segment text shown in a Mermaid node label. Enough to
#: recognise the segment, short enough that the diagram stays readable.
LABEL_CHARS = 40


def to_json(graph: DiscourseGraph) -> dict[str, Any]:
    return graph.model_dump()


def write_json(graph: DiscourseGraph, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(to_json(graph), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return p


def to_mermaid(graph: DiscourseGraph, label_chars: int = LABEL_CHARS) -> str:
    """A `flowchart TD` of the graph, one node per segment.

    Edge type is written on the arrow so a reader can tell a deterministic
    structural edge from an untyped `graft_dependency` at a glance.
    """
    lines = ["flowchart TD"]
    for seg in sorted(graph.segments, key=lambda s: s.order):
        lines.append(f'    {_node(seg.seg_id)}["{_label(seg.text, label_chars)}"]')
    for e in graph.edges:
        lines.append(f"    {_node(e.src)} -->|{e.type}| {_node(e.dst)}")
    # Forward references are drawn dotted: they are not edges, but seeing
    # where the corpus wanted one is the point of recording them.
    for r in graph.forward_refs:
        lines.append(f"    {_node(r.src)} -.->|forward_ref| {_node(r.dst)}")
    return "\n".join(lines) + "\n"


def write_mermaid(graph: DiscourseGraph, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_mermaid(graph), encoding="utf-8")
    return p


def write_graphml(graph: DiscourseGraph, path: str | Path) -> Path:
    """Write GraphML via networkx.

    GraphML cannot hold nulls, so `None` attributes are written as empty
    strings rather than omitted: a reader can then tell "no evidence" from
    "attribute missing".
    """
    import networkx as nx

    g = nx.DiGraph()
    for seg in sorted(graph.segments, key=lambda s: s.order):
        g.add_node(
            seg.seg_id,
            order=seg.order,
            seg_type=str(seg.seg_type),
            char_start=seg.char_start,
            char_end=seg.char_end,
            text=seg.text,
        )
    for e in graph.edges:
        g.add_edge(
            e.src,
            e.dst,
            type=e.type,
            origin=str(e.origin),
            evidence=e.evidence or "",
            confidence="" if e.confidence is None else float(e.confidence),
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, p)
    return p


def _node(seg_id: str) -> str:
    """Mermaid node ids cannot contain a hyphen, which every seg_id has."""
    return seg_id.replace("-", "_")


def _label(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    # Quotes and brackets would close the label early.
    return flat.replace('"', "'").replace("[", "(").replace("]", ")")

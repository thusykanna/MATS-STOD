"""Pass 1: edges the document's own layout already states. No LLM.

A heading governs the text beneath it, a lead-in sentence governs the list it
introduces, and a table header governs the cells in its column. None of that
needs a model to notice, and paying one to re-derive it would be both slower
and less reliable than reading it off the layout.

Nothing here fires under `PlainTextParser`, which types every block as a
level-0 paragraph (see D28): with no headings, list items or table cells
detected, each rule finds nothing to match. The code is written and tested
against hand-built `Document` objects so that it activates unchanged the day a
real layout parser lands, and so that the deferral is a statement about the
parser rather than about this pass.
"""

from __future__ import annotations

from ..schemas import Document, Edge, Segment


def _level_of(document: Document, segment: Segment) -> int:
    """Heading level of a segment, from the first block it covers."""
    for block_id in segment.block_ids:
        try:
            return document.block(block_id).level
        except KeyError:
            continue
    return 0


def heading_edges(document: Document, segments: list[Segment]) -> list[Edge]:
    """`structural_parent` from the nearest enclosing heading.

    A stack of open headings is kept rather than a scan backwards, so that a
    level-2 heading correctly reattaches to the level-1 heading above it once
    its own section closes.
    """
    out: list[Edge] = []
    stack: list[tuple[int, str]] = []
    for seg in sorted(segments, key=lambda s: s.order):
        if seg.seg_type == "heading":
            level = _level_of(document, seg)
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                out.append(_edge(stack[-1][1], seg.seg_id))
            stack.append((level, seg.seg_id))
        elif stack:
            out.append(_edge(stack[-1][1], seg.seg_id))
    return out


def list_lead_in_edges(segments: list[Segment]) -> list[Edge]:
    """A block ending in a colon governs the list items that follow it.

    The colon is the signal a lead-in gives: "the items below complete this
    sentence". Those items are usually unintelligible alone, which is exactly
    when the translator needs the lead-in as context.
    """
    ordered = sorted(segments, key=lambda s: s.order)
    out: list[Edge] = []
    for i, seg in enumerate(ordered):
        if seg.seg_type == "list_item" or not seg.text.rstrip().endswith(":"):
            continue
        for follower in ordered[i + 1 :]:
            if follower.seg_type != "list_item":
                break
            out.append(_edge(seg.seg_id, follower.seg_id))
    return out


def table_column_edges(segments: list[Segment]) -> list[Edge]:
    """A table's header cell governs the cells beneath it in the same column.

    Row and column live in segment metadata because no parser emits them yet;
    a segment without both is skipped rather than guessed at.
    """
    cells = [
        s
        for s in sorted(segments, key=lambda s: s.order)
        if s.seg_type == "table_cell" and "row" in s.metadata and "col" in s.metadata
    ]
    if not cells:
        return []
    header_row = min(int(c.metadata["row"]) for c in cells)
    headers = {
        int(c.metadata["col"]): c.seg_id
        for c in cells
        if int(c.metadata["row"]) == header_row
    }
    out: list[Edge] = []
    for cell in cells:
        if int(cell.metadata["row"]) == header_row:
            continue
        header = headers.get(int(cell.metadata["col"]))
        if header:
            out.append(_edge(header, cell.seg_id))
    return out


def infer_structural_edges(document: Document, segments: list[Segment]) -> list[Edge]:
    """Every deterministic edge the layout states, in one list."""
    return [
        *heading_edges(document, segments),
        *list_lead_in_edges(segments),
        *table_column_edges(segments),
    ]


def _edge(src: str, dst: str) -> Edge:
    return Edge(src=src, dst=dst, type="structural_parent", origin="structural", evidence=None)

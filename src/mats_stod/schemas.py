"""Core data models for MATS-STOD.

All intermediate artifacts are these models serialised to JSON, so that any
stage can be inspected or re-run in isolation without re-running the ones
before it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------

BlockType = Literal["heading", "paragraph", "list_item", "table_cell", "caption", "other"]

#: Canonical edge types. The set is closed but lives in config as well, so an
#: experiment can narrow it; `graph.edge_types` in the YAML is the authority at
#: run time and is validated against this list.
EDGE_TYPES: tuple[str, ...] = (
    "structural_parent",
    "coreference",
    "entity_introduction",
    "term_definition",
    "cross_reference",
    "continuation",
    # GRAFT's edge agent answers a single yes/no question and therefore cannot
    # name a relation; its edges carry this untyped label.
    "graft_dependency",
)

EdgeOrigin = Literal["structural", "llm"]


class ValidationErrorCode(str, Enum):  # noqa: UP042 - StrEnum changes JSON output
    """Machine-readable reasons a DiscourseGraph can be invalid."""

    OVERLAP = "segments_overlap"
    ORDER = "segments_out_of_order"
    COVERAGE = "segment_text_mismatch"
    DUPLICATE_ORDER = "duplicate_order"
    DUPLICATE_SEG_ID = "duplicate_seg_id"
    MISSING_ENDPOINT = "edge_endpoint_missing"
    SELF_LOOP = "edge_self_loop"
    DUPLICATE_EDGE = "duplicate_edge"
    BACKWARD_EDGE = "edge_not_forward"
    UNKNOWN_EDGE_TYPE = "unknown_edge_type"


class GraphValidationError(Exception):
    """Raised by DiscourseGraph.validate() when an invariant is broken."""

    def __init__(self, problems: list[tuple[ValidationErrorCode, str]]) -> None:
        self.problems = problems
        detail = "; ".join(f"{code.value}: {msg}" for code, msg in problems)
        super().__init__(detail)


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class LayoutBlock(BaseModel):
    """A physical region of the source document.

    Offsets index the NFC-normalised `Document.raw_text`, never a cleaned or
    re-flowed copy, so that every downstream artifact can be traced back to the
    exact source characters.
    """

    block_id: str
    type: BlockType = "paragraph"
    level: int = 0
    char_start: int
    char_end: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def text_of(self, raw_text: str) -> str:
        return raw_text[self.char_start : self.char_end]


class Document(BaseModel):
    doc_id: str
    source_lang: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocks: list[LayoutBlock] = Field(default_factory=list)

    def block(self, block_id: str) -> LayoutBlock:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        raise KeyError(block_id)


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------


class Segment(BaseModel):
    seg_id: str
    doc_id: str
    order: int
    text: str
    char_start: int
    char_end: int
    block_ids: list[str] = Field(default_factory=list)
    seg_type: BlockType = "paragraph"
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_span(self) -> Segment:
        if self.char_end < self.char_start:
            raise ValueError(f"segment {self.seg_id}: char_end < char_start")
        return self


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


class Edge(BaseModel):
    """A dependency: `dst` needs `src` to be translated well.

    `src` is always earlier in reading order than `dst`, which is what makes the
    assembled graph acyclic without a cycle check.
    """

    src: str
    dst: str
    type: str
    evidence: str | None = None
    origin: EdgeOrigin = "llm"
    confidence: float | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.type)


class ForwardRef(BaseModel):
    """A dependency pointing forward in reading order.

    These are recorded rather than dropped silently, because how often they
    occur is a property of the corpus worth reporting, but they are not edges:
    admitting them would allow cycles.
    """

    src: str
    dst: str
    type: str
    evidence: str | None = None
    origin: EdgeOrigin = "llm"
    confidence: float | None = None


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


class DiscourseGraph(BaseModel):
    doc_id: str
    segments: list[Segment] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    forward_refs: list[ForwardRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- lookups ----------------------------------------------------------

    def segment(self, seg_id: str) -> Segment:
        for s in self.segments:
            if s.seg_id == seg_id:
                return s
        raise KeyError(seg_id)

    def order_of(self, seg_id: str) -> int:
        return self.segment(seg_id).order

    def parents(self, seg_id: str) -> list[str]:
        """Segment ids that `seg_id` directly depends on, in reading order."""
        ids = [e.src for e in self.edges if e.dst == seg_id]
        return sorted(set(ids), key=self.order_of)

    def children(self, seg_id: str) -> list[str]:
        ids = [e.dst for e in self.edges if e.src == seg_id]
        return sorted(set(ids), key=self.order_of)

    def ancestors(self, seg_id: str, max_depth: int = 1) -> list[str]:
        """Transitive parents up to `max_depth` hops, in reading order.

        Depth is bounded because the context budget is bounded; an unbounded
        walk on a dense graph pulls in the whole document and silently turns
        D1 into B0.
        """
        if max_depth < 1:
            return []
        seen: set[str] = set()
        frontier = [seg_id]
        for _ in range(max_depth):
            nxt: list[str] = []
            for node in frontier:
                for p in self.parents(node):
                    if p not in seen and p != seg_id:
                        seen.add(p)
                        nxt.append(p)
            frontier = nxt
            if not frontier:
                break
        return sorted(seen, key=self.order_of)

    def topological_order(self) -> list[str]:
        """Kahn's algorithm, breaking ties by reading order.

        Ties are broken deterministically so two runs over the same graph
        produce the same translation order and therefore the same cache keys.
        """
        indeg: dict[str, int] = {s.seg_id: 0 for s in self.segments}
        adj: dict[str, list[str]] = defaultdict(list)
        seen_pairs: set[tuple[str, str]] = set()
        for e in self.edges:
            if (e.src, e.dst) in seen_pairs:
                continue
            seen_pairs.add((e.src, e.dst))
            adj[e.src].append(e.dst)
            indeg[e.dst] += 1

        ready = deque(
            sorted((sid for sid, d in indeg.items() if d == 0), key=self.order_of)
        )
        out: list[str] = []
        while ready:
            node = ready.popleft()
            out.append(node)
            newly: list[str] = []
            for child in adj[node]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    newly.append(child)
            if newly:
                merged = sorted(list(ready) + newly, key=self.order_of)
                ready = deque(merged)
        if len(out) != len(self.segments):
            raise GraphValidationError(
                [(ValidationErrorCode.BACKWARD_EDGE, "graph contains a cycle")]
            )
        return out

    # -- invariants -------------------------------------------------------

    def validate_graph(
        self,
        raw_text: str | None = None,
        allowed_edge_types: tuple[str, ...] | list[str] = EDGE_TYPES,
    ) -> None:
        """Enforce the three invariants. Raises GraphValidationError.

        `raw_text` is optional because a graph can be checked structurally
        without the document at hand, but when it is supplied, invariant 1 is
        checked against the actual characters rather than trusted.
        """
        problems: list[tuple[ValidationErrorCode, str]] = []
        problems += self._check_segments(raw_text)
        problems += self._check_edges(allowed_edge_types)
        if problems:
            raise GraphValidationError(problems)

    def _check_segments(self, raw_text: str | None) -> list[tuple[ValidationErrorCode, str]]:
        problems: list[tuple[ValidationErrorCode, str]] = []
        seen_ids: set[str] = set()
        seen_orders: set[int] = set()
        for s in self.segments:
            if s.seg_id in seen_ids:
                problems.append((ValidationErrorCode.DUPLICATE_SEG_ID, s.seg_id))
            seen_ids.add(s.seg_id)
            if s.order in seen_orders:
                problems.append((ValidationErrorCode.DUPLICATE_ORDER, f"order {s.order}"))
            seen_orders.add(s.order)

        ordered = sorted(self.segments, key=lambda s: s.order)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.char_start < a.char_end:
                problems.append(
                    (
                        ValidationErrorCode.OVERLAP,
                        f"{a.seg_id}[{a.char_start}:{a.char_end}] overlaps "
                        f"{b.seg_id}[{b.char_start}:{b.char_end}]",
                    )
                )
            if b.char_start < a.char_start:
                problems.append(
                    (ValidationErrorCode.ORDER, f"{b.seg_id} starts before {a.seg_id}")
                )

        if raw_text is not None:
            for s in ordered:
                actual = raw_text[s.char_start : s.char_end]
                if actual != s.text:
                    problems.append(
                        (
                            ValidationErrorCode.COVERAGE,
                            f"{s.seg_id}: stored text differs from raw_text slice",
                        )
                    )
            # Invariant 1: concatenating by offsets reproduces raw_text apart
            # from inter-segment whitespace.
            rebuilt = "".join(raw_text[s.char_start : s.char_end] for s in ordered)
            if _strip_ws(rebuilt) != _strip_ws(raw_text):
                problems.append(
                    (
                        ValidationErrorCode.COVERAGE,
                        "segments do not reproduce raw_text (non-whitespace loss)",
                    )
                )
        return problems

    def _check_edges(
        self, allowed_edge_types: tuple[str, ...] | list[str]
    ) -> list[tuple[ValidationErrorCode, str]]:
        problems: list[tuple[ValidationErrorCode, str]] = []
        orders = {s.seg_id: s.order for s in self.segments}
        seen: set[tuple[str, str, str]] = set()
        for e in self.edges:
            if e.src not in orders:
                problems.append((ValidationErrorCode.MISSING_ENDPOINT, f"src {e.src}"))
                continue
            if e.dst not in orders:
                problems.append((ValidationErrorCode.MISSING_ENDPOINT, f"dst {e.dst}"))
                continue
            if e.src == e.dst:
                problems.append((ValidationErrorCode.SELF_LOOP, e.src))
            if e.key() in seen:
                problems.append((ValidationErrorCode.DUPLICATE_EDGE, str(e.key())))
            seen.add(e.key())
            if orders[e.src] >= orders[e.dst]:
                problems.append(
                    (
                        ValidationErrorCode.BACKWARD_EDGE,
                        f"{e.src}(order {orders[e.src]}) -> {e.dst}(order {orders[e.dst]})",
                    )
                )
            if e.type not in allowed_edge_types:
                problems.append((ValidationErrorCode.UNKNOWN_EDGE_TYPE, e.type))
        return problems


def _strip_ws(text: str) -> str:
    """Drop whitespace for coverage comparison.

    Zero-width joiner (U+200D) is NOT whitespace and is deliberately kept: it
    is a letter-forming character in Sinhala and losing it changes the word.
    """
    return "".join(ch for ch in text if not ch.isspace())


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------


class TranslationRecord(BaseModel):
    seg_id: str
    source_text: str
    target_text: str
    context_strategy: str
    context_seg_ids: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    latency_s: float = 0.0
    flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

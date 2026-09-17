"""The three graph invariants, each tested by breaking it."""

from __future__ import annotations

import pytest

from mats_stod.schemas import (
    DiscourseGraph,
    Edge,
    GraphValidationError,
    Segment,
    ValidationErrorCode,
)


def seg(order: int, start: int, end: int, text: str) -> Segment:
    return Segment(
        seg_id=f"d-s{order:04d}", doc_id="d", order=order, text=text,
        char_start=start, char_end=end,
    )


RAW = "First part. Second part. Third part."


def good_graph() -> DiscourseGraph:
    return DiscourseGraph(
        doc_id="d",
        segments=[seg(0, 0, 11, "First part."), seg(1, 12, 24, "Second part."), seg(2, 25, 36, "Third part.")],
    )


def codes(exc: GraphValidationError) -> set[ValidationErrorCode]:
    return {c for c, _ in exc.problems}


def test_valid_graph_passes():
    good_graph().validate_graph(raw_text=RAW)


def test_overlapping_segments_rejected():
    g = good_graph()
    g.segments[1] = seg(1, 5, 24, RAW[5:24])
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.OVERLAP in codes(e.value)


def test_segment_text_must_match_offsets():
    g = good_graph()
    g.segments[0].text = "Something else"
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.COVERAGE in codes(e.value)


def test_dropped_characters_rejected():
    """Invariant 1: only inter-segment whitespace may be lost."""
    g = DiscourseGraph(doc_id="d", segments=[seg(0, 0, 11, RAW[0:11])])
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.COVERAGE in codes(e.value)


def test_backward_edge_rejected():
    g = good_graph()
    g.edges = [Edge(src="d-s0002", dst="d-s0000", type="continuation")]
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.BACKWARD_EDGE in codes(e.value)


def test_self_loop_rejected():
    g = good_graph()
    g.edges = [Edge(src="d-s0001", dst="d-s0001", type="continuation")]
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.SELF_LOOP in codes(e.value)


def test_duplicate_edge_rejected():
    g = good_graph()
    e = Edge(src="d-s0000", dst="d-s0001", type="continuation")
    g.edges = [e, e.model_copy()]
    with pytest.raises(GraphValidationError) as exc:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.DUPLICATE_EDGE in codes(exc.value)


def test_missing_endpoint_rejected():
    g = good_graph()
    g.edges = [Edge(src="d-s0000", dst="nope", type="continuation")]
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.MISSING_ENDPOINT in codes(e.value)


def test_unknown_edge_type_rejected():
    g = good_graph()
    g.edges = [Edge(src="d-s0000", dst="d-s0001", type="invented")]
    with pytest.raises(GraphValidationError) as e:
        g.validate_graph(raw_text=RAW)
    assert ValidationErrorCode.UNKNOWN_EDGE_TYPE in codes(e.value)


def test_parents_ancestors_and_topological_order():
    g = good_graph()
    g.edges = [
        Edge(src="d-s0000", dst="d-s0001", type="continuation"),
        Edge(src="d-s0001", dst="d-s0002", type="continuation"),
    ]
    g.validate_graph(raw_text=RAW)
    assert g.parents("d-s0002") == ["d-s0001"]
    assert g.ancestors("d-s0002", max_depth=1) == ["d-s0001"]
    assert g.ancestors("d-s0002", max_depth=2) == ["d-s0000", "d-s0001"]
    assert g.topological_order() == ["d-s0000", "d-s0001", "d-s0002"]


def test_topological_order_is_deterministic():
    """Ties break by reading order, so cache keys stay stable across runs."""
    g = good_graph()
    g.edges = [Edge(src="d-s0000", dst="d-s0002", type="continuation")]
    assert g.topological_order() == g.topological_order() == ["d-s0000", "d-s0001", "d-s0002"]

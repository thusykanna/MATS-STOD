"""The shared graph core: structural edges, assembly, stats, export, scoring.

Nothing here needs an LLM or a network; `graft_edges.py`'s own tests cover the
LLM-backed inferrer itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mats_stod.config import Settings
from mats_stod.evaluation.edge_score import (
    aggregate,
    load_gold_edges,
    prf,
    score_edges,
)
from mats_stod.graph.assemble import assemble_graph
from mats_stod.graph.export import to_mermaid, write_graphml, write_json
from mats_stod.graph.stats import aggregate_graph_stats, graph_stats
from mats_stod.graph.structural_edges import (
    heading_edges,
    infer_structural_edges,
    list_lead_in_edges,
    table_column_edges,
)
from mats_stod.schemas import (
    DiscourseGraph,
    Document,
    Edge,
    GraphValidationError,
    LayoutBlock,
    Segment,
)
from tests.conftest import SAMPLES

# --------------------------------------------------------------------------
# Fixtures built by hand, because no parser emits typed blocks yet (D28).
# --------------------------------------------------------------------------


def build_doc(parts: list[tuple[str, str, int]]) -> tuple[Document, list[Segment]]:
    """Build a Document and one Segment per part, blocks separated by a blank line.

    `parts` is a list of (text, block type, level). Offsets are computed from
    the assembled text so the segments satisfy invariant 1 for real.
    """
    raw_parts, offsets, cursor = [], [], 0
    for text, _, _ in parts:
        offsets.append((cursor, cursor + len(text)))
        raw_parts.append(text)
        cursor += len(text) + 2
    raw = "\n\n".join(raw_parts)

    blocks, segments = [], []
    for i, ((text, btype, level), (start, end)) in enumerate(zip(parts, offsets, strict=True)):
        blocks.append(
            LayoutBlock(block_id=f"b{i}", type=btype, level=level, char_start=start, char_end=end)
        )
        segments.append(
            Segment(
                seg_id=f"d-s{i:04d}",
                doc_id="d",
                order=i,
                text=text,
                char_start=start,
                char_end=end,
                block_ids=[f"b{i}"],
                seg_type=btype,
            )
        )
    return Document(doc_id="d", source_lang="si", raw_text=raw, blocks=blocks), segments


def plain_segments(n: int) -> list[Segment]:
    """`n` paragraph segments with no layout, for assembly and stats tests."""
    return [
        Segment(
            seg_id=f"d-s{i:04d}",
            doc_id="d",
            order=i,
            text=f"segment {i}",
            char_start=i * 12,
            char_end=i * 12 + 10,
        )
        for i in range(n)
    ]


def edge(src: int, dst: int, type_: str = "graft_dependency") -> Edge:
    return Edge(src=f"d-s{src:04d}", dst=f"d-s{dst:04d}", type=type_)


# --------------------------------------------------------------------------
# Pass 1: structural edges
# --------------------------------------------------------------------------


def test_heading_governs_following_paragraph() -> None:
    doc, segs = build_doc(
        [("1. Introduction", "heading", 1), ("This circular applies to all.", "paragraph", 0)]
    )
    edges = heading_edges(doc, segs)
    assert [(e.src, e.dst) for e in edges] == [("d-s0000", "d-s0001")]
    assert edges[0].type == "structural_parent"
    assert edges[0].origin == "structural"


def test_subheading_attaches_to_its_parent_heading() -> None:
    """A level-2 heading hangs off the level-1 above it, not off a sibling."""
    doc, segs = build_doc(
        [
            ("1. Scope", "heading", 1),
            ("1.1 Travel", "heading", 2),
            ("Travel is included.", "paragraph", 0),
            ("2. Limits", "heading", 1),
            ("The maximum is revised.", "paragraph", 0),
        ]
    )
    pairs = {(e.src, e.dst) for e in heading_edges(doc, segs)}
    assert ("d-s0000", "d-s0001") in pairs  # 1.1 under 1
    assert ("d-s0001", "d-s0002") in pairs  # paragraph under 1.1
    assert ("d-s0003", "d-s0004") in pairs  # paragraph under 2
    # Section 2 closed section 1, so it must not hang off 1.1.
    assert ("d-s0001", "d-s0003") not in pairs


def test_colon_lead_in_governs_the_list_that_follows() -> None:
    _, segs = build_doc(
        [
            ("Submit the following documents:", "paragraph", 0),
            ("Approved application", "list_item", 0),
            ("Original receipts", "list_item", 0),
            ("Payment limits apply.", "paragraph", 0),
        ]
    )
    pairs = {(e.src, e.dst) for e in list_lead_in_edges(segs)}
    assert pairs == {("d-s0000", "d-s0001"), ("d-s0000", "d-s0002")}


def test_table_header_governs_its_column() -> None:
    segs = [
        Segment(
            seg_id=f"d-s{i:04d}",
            doc_id="d",
            order=i,
            text=t,
            char_start=i * 10,
            char_end=i * 10 + len(t),
            seg_type="table_cell",
            metadata={"row": r, "col": c},
        )
        for i, (t, r, c) in enumerate(
            [("Grade", 0, 0), ("Rate", 0, 1), ("I", 1, 0), ("3500", 1, 1)]
        )
    ]
    pairs = {(e.src, e.dst) for e in table_column_edges(segs)}
    assert pairs == {("d-s0000", "d-s0002"), ("d-s0001", "d-s0003")}


def test_structural_pass_is_empty_under_the_plaintext_parser(sample_doc, settings) -> None:
    """D28 in the flesh: no typed blocks, so pass 1 finds nothing.

    This is the documented consequence of deferring layout extraction. If a
    real parser lands and this test starts failing, that is the signal the
    consequence no longer holds and DECISIONS.md needs revising.
    """
    from mats_stod.llm.factory import build_llm
    from mats_stod.llm.fake import FakeLLM
    from mats_stod.segmentation.graft_discourse import GraftDiscourseSegmenter

    settings.llm.use_cache = False
    llm = build_llm(settings, provider=FakeLLM(default="no"))
    segs = GraftDiscourseSegmenter(settings, llm).segment(sample_doc).segments
    assert infer_structural_edges(sample_doc, segs) == []


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def test_assembly_keeps_forward_edges_and_validates(settings: Settings) -> None:
    graph, stats = assemble_graph("d", plain_segments(4), [edge(0, 1), edge(0, 3)], settings)
    assert {(e.src, e.dst) for e in graph.edges} == {("d-s0000", "d-s0001"), ("d-s0000", "d-s0003")}
    assert stats.n_kept == 2
    graph.validate_graph(allowed_edge_types=settings.graph.edge_types)


def test_backward_edge_becomes_a_forward_ref_not_an_edge(settings: Settings) -> None:
    """A dependency on a later segment is recorded, never admitted.

    Admitting one would allow a cycle, and a cycle makes topological_order
    raise, so the whole pipeline would fail on one bad LLM answer.
    """
    graph, stats = assemble_graph("d", plain_segments(3), [edge(2, 0)], settings)
    assert graph.edges == []
    assert len(graph.forward_refs) == 1
    assert (graph.forward_refs[0].src, graph.forward_refs[0].dst) == ("d-s0002", "d-s0000")
    assert stats.n_forward_refs == 1


def test_self_loop_and_unknown_endpoint_are_dropped_and_counted(settings: Settings) -> None:
    proposed = [
        Edge(src="d-s0001", dst="d-s0001", type="graft_dependency"),
        Edge(src="d-s0000", dst="d-s9999", type="graft_dependency"),
        edge(0, 1),
    ]
    graph, stats = assemble_graph("d", plain_segments(3), proposed, settings)
    assert stats.dropped_self_loop == 1
    assert stats.dropped_unknown_endpoint == 1
    assert len(graph.edges) == 1


def test_duplicate_edges_collapse(settings: Settings) -> None:
    graph, stats = assemble_graph("d", plain_segments(3), [edge(0, 1), edge(0, 1)], settings)
    assert len(graph.edges) == 1
    assert stats.dropped_duplicate == 1


def test_same_pair_with_two_types_is_not_a_duplicate(settings: Settings) -> None:
    """Dedupe is on (src, dst, type): two relations between one pair are two edges."""
    proposed = [edge(0, 1, "continuation"), edge(0, 1, "graft_dependency")]
    graph, stats = assemble_graph("d", plain_segments(3), proposed, settings)
    assert len(graph.edges) == 2
    assert stats.dropped_duplicate == 0


def test_parent_cap_keeps_the_nearest_parents(settings: Settings) -> None:
    """With no confidence to rank by, nearest-in-reading-order wins (D: parent cap)."""
    settings.graph.max_parents = 2
    proposed = [edge(i, 5) for i in range(5)]
    graph, stats = assemble_graph("d", plain_segments(6), proposed, settings)
    assert set(graph.parents("d-s0005")) == {"d-s0003", "d-s0004"}
    assert stats.dropped_parent_cap == 3
    assert stats.parent_cap_rule == "nearest_by_order"


def test_transitive_reduction_is_off_by_default_and_removes_implied_edges(
    settings: Settings,
) -> None:
    proposed = [edge(0, 1), edge(1, 2), edge(0, 2)]
    kept, _ = assemble_graph("d", plain_segments(3), proposed, settings)
    assert len(kept.edges) == 3

    settings.graph.transitive_reduction = True
    reduced, stats = assemble_graph("d", plain_segments(3), proposed, settings)
    assert {(e.src, e.dst) for e in reduced.edges} == {
        ("d-s0000", "d-s0001"),
        ("d-s0001", "d-s0002"),
    }
    assert stats.dropped_transitive == 1


def test_assembly_rejects_an_edge_type_outside_the_configured_set(settings: Settings) -> None:
    settings.graph.edge_types = ["continuation"]
    with pytest.raises(GraphValidationError):
        assemble_graph("d", plain_segments(3), [edge(0, 1, "graft_dependency")], settings)


def test_assembly_checks_segment_text_against_the_document(settings: Settings) -> None:
    """Invariant 1 is enforced with the real characters when raw_text is given."""
    segs = plain_segments(2)
    segs[0] = segs[0].model_copy(update={"text": "not what the document says"})
    with pytest.raises(GraphValidationError):
        assemble_graph("d", segs, [], settings, raw_text="segment 0  segment 1")


def test_assembly_stats_are_written_into_graph_metadata(settings: Settings) -> None:
    """Nothing is discarded quietly: the counts travel with the graph."""
    graph, _ = assemble_graph("d", plain_segments(3), [edge(0, 1), edge(0, 1)], settings)
    assert graph.metadata["assembly"]["dropped_duplicate"] == 1
    assert graph.metadata["assembly"]["n_proposed"] == 2


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def test_adjacent_only_share_is_one_for_a_chain(settings: Settings) -> None:
    """A predecessor-shaped graph says nothing a sliding window did not."""
    graph, _ = assemble_graph("d", plain_segments(4), [edge(0, 1), edge(1, 2), edge(2, 3)], settings)
    stats = graph_stats(graph)
    assert stats["adjacent_only_share"] == 1.0
    assert stats["segments_with_nonadjacent_parent"] == 0


def test_nonadjacent_edges_are_counted_and_name_the_m4_subset(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(5), [edge(0, 1), edge(0, 4)], settings)
    stats = graph_stats(graph)
    assert stats["n_nonadjacent_edges"] == 1
    assert stats["adjacent_only_share"] == 0.5
    assert stats["segments_with_nonadjacent_parent"] == 1  # only s0004
    assert stats["edge_span_max"] == 4


def test_stats_report_depth_and_isolated_segments(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(4), [edge(0, 1), edge(1, 2)], settings)
    stats = graph_stats(graph)
    assert stats["depth"] == 2
    assert stats["n_isolated_segments"] == 1  # s0003 has no parent and no child
    assert stats["in_degree_max"] == 1


def test_empty_graph_stats_do_not_divide_by_zero(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(3), [], settings)
    stats = graph_stats(graph)
    assert stats["n_edges"] == 0
    assert stats["adjacent_only_share"] == 0.0
    assert stats["depth"] == 0


def test_corpus_aggregate_pools_edges_not_document_means(settings: Settings) -> None:
    a, _ = assemble_graph("d", plain_segments(3), [edge(0, 1)], settings)
    b, _ = assemble_graph("d", plain_segments(4), [edge(0, 3)], settings)
    agg = aggregate_graph_stats([graph_stats(a), graph_stats(b)])
    assert agg["n_documents"] == 2
    assert agg["n_edges_total"] == 2
    assert agg["adjacent_only_share"] == 0.5


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_mermaid_escapes_ids_and_labels(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(2), [edge(0, 1)], settings)
    text = to_mermaid(graph)
    assert text.startswith("flowchart TD")
    assert "d_s0000 -->|graft_dependency| d_s0001" in text
    assert "-" not in text.split("\n")[1].split("[")[0]  # node ids carry no hyphen


def test_forward_refs_are_drawn_dotted(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(3), [edge(2, 0)], settings)
    assert "-.->|forward_ref|" in to_mermaid(graph)


def test_exports_are_written_and_reload(settings: Settings, tmp_path: Path) -> None:
    graph, _ = assemble_graph("d", plain_segments(3), [edge(0, 2)], settings)
    p = write_json(graph, tmp_path / "graph.json")
    reloaded = DiscourseGraph.model_validate(json.loads(p.read_text(encoding="utf-8")))
    assert len(reloaded.edges) == 1

    import networkx as nx

    gml = write_graphml(graph, tmp_path / "graph.graphml")
    back = nx.read_graphml(gml)
    assert back.number_of_nodes() == 3
    assert back.number_of_edges() == 1


# --------------------------------------------------------------------------
# Edge scoring
# --------------------------------------------------------------------------


def test_prf_on_empty_sets_is_perfect_agreement() -> None:
    assert prf(set(), set()) == (1.0, 1.0, 1.0)


def test_perfect_prediction_scores_one(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(3), [edge(0, 2, "coreference")], settings)
    gold = [{"src_segment_index": 0, "dst_segment_index": 2, "type": "coreference"}]
    score = score_edges("d", graph, gold)
    assert score.f1 == 1.0
    assert score.f1_untyped == 1.0


def test_untyped_score_rescues_an_untyped_inferrer(settings: Settings) -> None:
    """GRAFT finds the right pair but cannot name the relation (D6).

    Typed scoring would report zero for a method that located the dependency
    correctly, which is why both numbers are reported.
    """
    graph, _ = assemble_graph("d", plain_segments(3), [edge(0, 2, "graft_dependency")], settings)
    gold = [{"src_segment_index": 0, "dst_segment_index": 2, "type": "coreference"}]
    score = score_edges("d", graph, gold)
    assert score.f1 == 0.0
    assert score.f1_untyped == 1.0


def test_missed_and_spurious_edges_move_recall_and_precision(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(4), [edge(0, 1), edge(0, 3)], settings)
    gold = [
        {"src_segment_index": 0, "dst_segment_index": 1, "type": "continuation"},
        {"src_segment_index": 1, "dst_segment_index": 2, "type": "continuation"},
    ]
    score = score_edges("d", graph, gold)
    assert score.precision_untyped == 0.5  # one of two predictions is right
    assert score.recall_untyped == 0.5  # one of two gold edges found


def test_gold_edges_load_from_the_annotate_import_format(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "doc_id": "d",
                "source_lang": "si",
                "edges": [
                    {
                        "src_segment_index": 0,
                        "dst_segment_index": 2,
                        "src_unit": "u1",
                        "dst_unit": "u3",
                        "type": "cross_reference",
                        "evidence": "the said circular",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_gold_edges(path)[0]["type"] == "cross_reference"


def test_aggregate_over_documents(settings: Settings) -> None:
    graph, _ = assemble_graph("d", plain_segments(3), [edge(0, 2, "coreference")], settings)
    gold = [{"src_segment_index": 0, "dst_segment_index": 2, "type": "coreference"}]
    scores = [score_edges("a", graph, gold), score_edges("b", graph, [])]
    agg = aggregate(scores)
    assert agg["n_documents"] == 2
    assert agg["f1_untyped"] == 0.5  # perfect on one, zero on the other


# --------------------------------------------------------------------------
# Pipeline and CLI
# --------------------------------------------------------------------------


def _fake_graph_llm(settings: Settings):
    from mats_stod.llm.factory import build_llm
    from mats_stod.llm.fake import FakeLLM

    settings.llm.use_cache = False
    # Every yes/no decision comes back "no", so the graph carries only the
    # unconditional predecessor edges GRAFT always adds, letting these tests
    # stay deterministic and offline.
    return build_llm(settings, provider=FakeLLM(default="no"))


def test_build_graph_pipeline_writes_every_artifact(settings: Settings, tmp_path: Path) -> None:
    """The whole path runs end to end against a fake, offline LLM."""
    from mats_stod.io.parallel import load_parallel
    from mats_stod.io.runs import RunDir
    from mats_stod.pipelines.graph_pipeline import run_graph_build

    llm = _fake_graph_llm(settings)
    pairs = load_parallel(SAMPLES, "si", "ta", max_docs=1)
    run = RunDir.create(settings, prefix="test-graph")
    extra = run_graph_build(pairs, settings, llm, run)

    base = run.path / "documents" / pairs[0].doc_id
    for name in ("graph.json", "graph.mmd", "graph.graphml", "graph_stats.json"):
        assert (base / name).exists(), name
    assert extra["graph_stats"]["n_documents"] == 1
    # No gold edges in a temp dir, so the report must say so rather than
    # reporting a score of zero.
    assert "notes" in extra


def test_build_graph_scores_against_gold_when_it_exists(settings: Settings) -> None:
    from mats_stod.evaluation.annotate import gold_stem
    from mats_stod.io.parallel import load_parallel
    from mats_stod.io.runs import RunDir
    from mats_stod.pipelines.graph_pipeline import build_document_graph, run_graph_build

    llm = _fake_graph_llm(settings)
    pairs = load_parallel(SAMPLES, "si", "ta", max_docs=1)
    built = build_document_graph(pairs[0], settings, llm)

    gold_dir = Path(settings.paths.gold_edges)
    gold_dir.mkdir(parents=True, exist_ok=True)
    stem = gold_stem(built.doc_id, built.source_lang)
    (gold_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "doc_id": built.doc_id,
                "source_lang": built.source_lang,
                "edges": [
                    {"src_segment_index": 0, "dst_segment_index": 2, "type": "cross_reference"}
                ],
            }
        ),
        encoding="utf-8",
    )

    extra = run_graph_build(pairs, settings, llm, RunDir.create(settings, prefix="test-graph"))
    assert "edge_scores" in extra
    # Every decision came back "no", so recall is zero against a real gold edge.
    assert extra["edge_scores"]["corpus"]["recall_untyped"] == 0.0


def test_cli_build_graph_runs_offline() -> None:
    from typer.testing import CliRunner

    from mats_stod.cli import app

    result = CliRunner().invoke(
        app,
        [
            "build-graph",
            "--provider",
            "fake",
            "--data",
            str(SAMPLES),
            "--portion",
            "all",
            "--max-docs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "adjacent-only edges" in result.output

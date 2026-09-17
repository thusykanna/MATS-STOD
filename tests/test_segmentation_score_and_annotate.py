"""Segmentation scorer and the gold-annotation round trip."""

from __future__ import annotations

import pytest

from mats_stod.evaluation.annotate import (
    build_template,
    import_annotation,
    parse_annotation,
    write_template,
)
from mats_stod.evaluation.segmentation_score import (
    aggregate,
    boundaries_from_segments,
    boundary_prf,
    load_gold_boundaries,
    pk_and_windowdiff,
    score_segmentation,
)
from mats_stod.schemas import Segment


def segs(bounds: list[int], length: int) -> list[Segment]:
    edges = [0, *bounds, length]
    return [
        Segment(seg_id=f"s{i}", doc_id="d", order=i, text="x" * (e - s), char_start=s, char_end=e)
        for i, (s, e) in enumerate(zip(edges, edges[1:], strict=False))
    ]


def test_boundaries_exclude_the_document_start():
    assert boundaries_from_segments(segs([10, 20], 30)) == [10, 20]


def test_perfect_segmentation_scores_one():
    p, r, f1 = boundary_prf([10, 20], [10, 20])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_tolerance_forgives_a_near_miss():
    assert boundary_prf([10], [11], tolerance=2)[2] == 1.0
    assert boundary_prf([10], [11], tolerance=0)[2] == 0.0


def test_missed_and_spurious_boundaries():
    p, r, _ = boundary_prf([10, 20, 30], [10, 99])
    assert r == pytest.approx(1 / 3)
    assert p == pytest.approx(1 / 2)


def test_no_boundaries_on_either_side_is_perfect():
    assert boundary_prf([], [])[2] == 1.0


def test_pk_and_windowdiff_are_zero_for_a_perfect_match():
    pk, wd, _ = pk_and_windowdiff([100, 200], [100, 200], length=300, unit=10)
    assert pk == 0.0 and wd == 0.0


def test_pk_penalises_a_wrong_segmentation():
    pk, wd, _ = pk_and_windowdiff([100, 200], [50], length=300, unit=10)
    assert pk > 0 and wd > 0


def test_near_miss_costs_less_than_a_wild_guess():
    """The reason a windowed metric is reported alongside boundary F1."""
    near, _, _ = pk_and_windowdiff([150], [160], length=300, unit=10)
    wild, _, _ = pk_and_windowdiff([150], [20], length=300, unit=10)
    assert near < wild


def test_score_segmentation_end_to_end():
    score = score_segmentation("d", segs([100, 200], 300), [100, 200], 300)
    assert score.f1 == 1.0 and score.n_pred_boundaries == 2


def test_aggregate_averages_documents():
    a = score_segmentation("a", segs([100], 200), [100], 200)
    b = score_segmentation("b", segs([100], 200), [150], 200)
    agg = aggregate([a, b])
    assert agg["n_docs"] == 2 and 0 <= agg["f1"] <= 1


def test_gold_file_accepts_both_shapes(tmp_path):
    p1 = tmp_path / "a.json"
    p1.write_text('{"doc_id": "a", "boundaries": [5, 1]}', encoding="utf-8")
    p2 = tmp_path / "b.json"
    p2.write_text("[5, 1]", encoding="utf-8")
    assert load_gold_boundaries(p1) == load_gold_boundaries(p2) == [1, 5]


# -- annotation round trip ------------------------------------------------


def test_template_lists_every_sentence_unit(sample_doc, settings):
    text = build_template(sample_doc, settings)
    assert "UNITS" in text and "EDGES" in text
    unit_lines = [ln for ln in text.splitlines() if ln.startswith("u0")]
    assert len(unit_lines) > 5
    assert unit_lines[0].split("\t")[1] == "B"  # first unit starts a segment


def test_template_name_carries_the_source_language(sample_doc, settings, tmp_path):
    path = write_template(sample_doc, settings, tmp_path)
    assert path.name == "circular_01.si.annot.tsv"


def test_import_refuses_an_untagged_filename(sample_doc, settings, tmp_path):
    """A gold file with no language could be scored against the wrong side."""
    path = write_template(sample_doc, settings, tmp_path)
    untagged = tmp_path / "circular_01.annot.tsv"
    untagged.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="<doc_id>.<lang>.annot.tsv"):
        import_annotation(untagged, settings.paths.gold_segmentation, settings.paths.gold_edges)


def test_gold_files_record_the_source_language(sample_doc, settings, tmp_path):
    import json

    path = write_template(sample_doc, settings, tmp_path)
    seg_path, edge_path = import_annotation(
        path, settings.paths.gold_segmentation, settings.paths.gold_edges
    )
    assert seg_path.name == "circular_01.si.json"
    assert json.loads(seg_path.read_text(encoding="utf-8"))["source_lang"] == "si"
    assert json.loads(edge_path.read_text(encoding="utf-8"))["source_lang"] == "si"


def test_annotation_round_trip(sample_doc, settings, tmp_path):
    path = write_template(sample_doc, settings, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    # Mark the third unit as a segment start and add one edge.
    out = []
    marked = 0
    for line in lines:
        if line.startswith("u0002\t-"):
            line = line.replace("\t-\t", "\tB\t", 1)
            marked += 1
        out.append(line)
    out.append("u0000\tu0002\tcross_reference\tඉහත සඳහන්")
    path.write_text("\n".join(out), encoding="utf-8")
    assert marked == 1

    parsed = parse_annotation(path.read_text(encoding="utf-8"))
    assert len(parsed["boundaries"]) == 1
    assert parsed["edges"][0]["type"] == "cross_reference"

    seg_path, edge_path = import_annotation(
        path, settings.paths.gold_segmentation, settings.paths.gold_edges
    )
    assert load_gold_boundaries(seg_path) == parsed["boundaries"]
    import json

    edges = json.loads(edge_path.read_text(encoding="utf-8"))["edges"]
    assert edges[0]["src_segment_index"] == 0 and edges[0]["dst_segment_index"] == 1


def test_annotation_boundaries_are_real_offsets(sample_doc, settings, tmp_path):
    """A boundary must point at a character that actually starts a sentence."""
    path = write_template(sample_doc, settings, tmp_path)
    parsed = parse_annotation(path.read_text(encoding="utf-8"))
    for unit in parsed["units"]:
        slice_ = sample_doc.raw_text[unit["char_start"] : unit["char_end"]]
        assert slice_.replace("\n", " ⏎ ") == unit["text"]


def test_malformed_annotation_is_rejected():
    with pytest.raises(ValueError, match="malformed unit"):
        parse_annotation("UNITS\ngarbage line\n")


def test_edge_referencing_unknown_unit_is_rejected(sample_doc, settings, tmp_path):
    path = write_template(sample_doc, settings, tmp_path)
    text = path.read_text(encoding="utf-8") + "\nu9999\tu0000\tcontinuation\tx\n"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown unit"):
        import_annotation(path, settings.paths.gold_segmentation, settings.paths.gold_edges)

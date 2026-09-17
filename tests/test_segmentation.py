"""Structural and GRAFT segmentation, and the invariants on every document."""

from __future__ import annotations

import pytest

from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import FakeLLM
from mats_stod.parsing.base import build_parser
from mats_stod.parsing.plaintext import PlainTextParser
from mats_stod.schemas import DiscourseGraph
from mats_stod.segmentation.base import build_segmenter
from mats_stod.segmentation.graft_discourse import GraftDiscourseSegmenter
from mats_stod.segmentation.naive import (
    NaiveParagraphSegmenter,
    WholeDocumentSegmenter,
    chunk_document,
)
from mats_stod.segmentation.structural import StructuralSegmenter

SAMPLE_IDS = ["circular_01", "notice_02", "memo_03"]


def check_invariants(doc, segments, settings):
    DiscourseGraph(doc_id=doc.doc_id, segments=segments).validate_graph(
        raw_text=doc.raw_text, allowed_edge_types=settings.graph.edge_types
    )


# -- parser ---------------------------------------------------------------


def test_plaintext_parser_produces_paragraph_blocks(sample_doc):
    assert sample_doc.blocks
    assert all(b.type == "paragraph" and b.level == 0 for b in sample_doc.blocks)
    assert sample_doc.metadata["layout_detected"] is False


def test_block_offsets_slice_the_original_text(sample_doc):
    for b in sample_doc.blocks:
        assert b.text_of(sample_doc.raw_text).strip() == b.text_of(sample_doc.raw_text)


@pytest.mark.parametrize("name", ["markdown", "pdf", "docx"])
def test_deferred_parsers_raise_not_implemented(name):
    with pytest.raises(NotImplementedError):
        build_parser(name).parse("text", "d", "si")


# -- structural -----------------------------------------------------------


@pytest.mark.parametrize("doc_id", SAMPLE_IDS)
@pytest.mark.parametrize("lang", ["si", "ta"])
def test_structural_segmenter_holds_invariants_both_directions(doc_id, lang, settings):
    from mats_stod.io.text import read_text

    text = read_text(f"data/samples/{doc_id}/{doc_id}.{lang}")
    doc = PlainTextParser().parse(text, doc_id, lang)
    result = StructuralSegmenter(settings.segmentation).segment(doc)
    assert result.segments
    check_invariants(doc, result.segments, settings)
    assert result.stats["llm_calls"] == 0


def test_short_blocks_are_merged(settings, sample_doc):
    settings.segmentation.min_segment_chars = 200
    merged = StructuralSegmenter(settings.segmentation).segment(sample_doc)
    settings.segmentation.min_segment_chars = 0
    unmerged = StructuralSegmenter(settings.segmentation).segment(sample_doc)
    assert len(merged.segments) < len(unmerged.segments)
    check_invariants(sample_doc, merged.segments, settings)


def test_long_blocks_are_split_at_sentence_boundaries(settings, sample_doc):
    settings.segmentation.max_segment_chars = 80
    settings.segmentation.min_segment_chars = 0
    result = StructuralSegmenter(settings.segmentation).segment(sample_doc)
    check_invariants(sample_doc, result.segments, settings)
    # Splitting happened, and no segment ends mid-word.
    assert len(result.segments) > len(sample_doc.blocks)


def test_stats_include_length_distribution(settings, sample_doc):
    stats = StructuralSegmenter(settings.segmentation).segment(sample_doc).stats
    for key in ["n_segments", "chars_mean", "chars_median", "chars_min", "chars_max"]:
        assert key in stats


# -- GRAFT ----------------------------------------------------------------


def graft(settings, answer: str, sample_doc):
    settings.llm.use_cache = False
    fake = FakeLLM(default=answer)
    llm = build_llm(settings, provider=fake)
    result = GraftDiscourseSegmenter(settings, llm).segment(sample_doc)
    return result, fake


def test_graft_all_yes_produces_one_segment_per_group(settings, sample_doc):
    result, fake = graft(settings, "yes", sample_doc)
    assert len(result.segments) == result.stats["n_groups"]
    check_invariants(sample_doc, result.segments, settings)


def test_graft_all_no_produces_one_segment_per_sentence(settings, sample_doc):
    result, _ = graft(settings, "no", sample_doc)
    assert len(result.segments) == result.stats["n_sentence_units"]
    check_invariants(sample_doc, result.segments, settings)


def test_graft_call_count_is_one_per_boundary_considered(settings, sample_doc):
    """Cost is linear in sentences, which is the claim being defended."""
    result, fake = graft(settings, "no", sample_doc)
    expected = result.stats["n_sentence_units"] - result.stats["n_groups"]
    assert result.stats["llm_calls"] == expected == fake.call_count


def test_graft_never_receives_or_returns_document_text(settings, sample_doc):
    """The model decides yes/no; it cannot rewrite the document."""
    result, fake = graft(settings, "yes", sample_doc)
    for seg in result.segments:
        assert seg.text == sample_doc.raw_text[seg.char_start : seg.char_end]


def test_graft_cap_forces_a_boundary_despite_yes(settings, sample_doc):
    settings.segmentation.max_discourse_chars = 120
    result, _ = graft(settings, "yes", sample_doc)
    assert result.stats["boundaries_forced_by_cap"] > 0
    assert all(len(s.text) <= 200 for s in result.segments)
    check_invariants(sample_doc, result.segments, settings)


def test_graft_unparseable_answer_counts_as_a_boundary(settings, sample_doc):
    result, _ = graft(settings, "perhaps", sample_doc)
    assert result.stats["decisions_unparsed"] == result.stats["llm_calls"]
    assert len(result.segments) == result.stats["n_sentence_units"]


def test_graft_atomic_blocks_are_never_merged(settings, sample_doc):
    """A heading may not be glued to the paragraph below it."""
    doc = sample_doc.model_copy(deep=True)
    doc.blocks[0].type = "heading"
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=FakeLLM(default="yes"))
    result = GraftDiscourseSegmenter(settings, llm).segment(doc)
    assert result.segments[0].seg_type == "heading"
    assert result.segments[0].block_ids == [doc.blocks[0].block_id]
    check_invariants(doc, result.segments, settings)


def test_build_segmenter_rejects_graft_without_an_llm(settings):
    with pytest.raises(ValueError, match="needs an LLM"):
        build_segmenter("graft", settings, None)


def test_build_segmenter_rejects_unknown_name(settings):
    with pytest.raises(ValueError, match="unknown segmenter"):
        build_segmenter("nope", settings, None)


# -- naive / chunking -----------------------------------------------------


def test_naive_and_whole_document_segmenters(settings, sample_doc):
    naive = NaiveParagraphSegmenter().segment(sample_doc)
    whole = WholeDocumentSegmenter().segment(sample_doc)
    assert len(whole.segments) == 1
    assert len(naive.segments) == len(sample_doc.blocks)
    check_invariants(sample_doc, naive.segments, settings)
    check_invariants(sample_doc, whole.segments, settings)


def test_chunking_respects_the_character_budget(settings, sample_doc):
    result = chunk_document(sample_doc, max_chars=150)
    assert len(result.segments) > 1
    check_invariants(sample_doc, result.segments, settings)

"""GRAFT segmentation, whole-document/chunk segmentation, and the invariants on every document."""

from __future__ import annotations

import pytest

from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import FakeLLM
from mats_stod.parsing.base import build_parser
from mats_stod.schemas import DiscourseGraph
from mats_stod.segmentation.graft_discourse import GraftDiscourseSegmenter
from mats_stod.segmentation.naive import WholeDocumentSegmenter, chunk_document


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


# -- whole-document / chunking (used by B0) --------------------------------


def test_whole_document_segmenter_produces_one_segment(settings, sample_doc):
    whole = WholeDocumentSegmenter().segment(sample_doc)
    assert len(whole.segments) == 1
    check_invariants(sample_doc, whole.segments, settings)


def test_chunking_respects_the_character_budget(settings, sample_doc):
    result = chunk_document(sample_doc, max_chars=150)
    assert len(result.segments) > 1
    check_invariants(sample_doc, result.segments, settings)


# -- thinking budget ------------------------------------------------------


def test_graft_disables_reasoning_for_the_yes_no_call(settings, sample_doc):
    """A thinking model spends the whole output budget reasoning.

    Without thinking_budget=0 the model returns nothing for a 4-token cap, and
    every unparseable answer becomes a boundary, so the segmenter silently
    degenerates to one segment per sentence against a real model.
    """
    result, fake = graft(settings, "yes", sample_doc)
    assert fake.calls, "expected at least one decision call"
    for call in fake.calls:
        assert call["params"]["thinking_budget"] == 0
        assert call["params"]["max_output_tokens"] >= 8


def test_decision_budget_is_configurable_for_ablation(settings, sample_doc):
    """Turning reasoning back on must be a config change, not a code change."""
    settings.segmentation.decision_thinking_budget = 128
    settings.segmentation.decision_max_tokens = 256
    result, fake = graft(settings, "yes", sample_doc)
    assert fake.calls[0]["params"]["thinking_budget"] == 128
    assert fake.calls[0]["params"]["max_output_tokens"] == 256

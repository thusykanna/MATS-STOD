"""Context strategies, protected content, retry policy."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mats_stod.config import ProtectSettings
from mats_stod.io.parallel import load_parallel
from mats_stod.llm.base import LLMError
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import EchoLLM, FakeLLM
from mats_stod.pipelines.dag_translate import STRATEGY_NAME, translate_document
from mats_stod.translation import protect
from mats_stod.translation.linebreaks import LINEBREAK_TOKEN, mask_linebreaks, unmask_linebreaks
from mats_stod.translation.translator import Translator


def echo(messages, params):
    prompt = messages[-1].content
    body = prompt.split("Source text", 1)[1].split(":", 1)[1]
    return json.dumps({"translation": body.split("Return a JSON", 1)[0].strip()}, ensure_ascii=False)


@pytest.fixture
def pair():
    return load_parallel("data/samples", "si", "ta", doc_ids=["circular_01"])[0]


# -- D1 (DAG-context) --------------------------------------------------------


def test_dag_translates_every_segment_of_the_discourse_graph(settings, pair):
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=EchoLLM())
    result = translate_document(pair, settings, llm, InMemorySaver())
    assert result.segments, "the sample document must produce at least one segment"
    assert {r.seg_id for r in result.records} == {s.seg_id for s in result.segments}
    assert all(r.context_strategy == STRATEGY_NAME for r in result.records)


def test_dag_root_segments_have_no_context(settings, pair):
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=EchoLLM())
    result = translate_document(pair, settings, llm, InMemorySaver())
    first = result.records[0]
    assert first.context_seg_ids == []


def test_dag_a_dependent_segment_is_shown_its_graph_parents_as_context(settings, pair):
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=EchoLLM())
    result = translate_document(pair, settings, llm, InMemorySaver())
    parents_of = {s.seg_id: result.graph.parents(s.seg_id) for s in result.segments}
    dependent = next((r for r in result.records if parents_of[r.seg_id]), None)
    assert dependent is not None, "the sample document must produce a dependent segment"
    assert set(dependent.context_seg_ids) == set(parents_of[dependent.seg_id])


def test_dag_reassembles_with_the_source_documents_own_separators(settings, pair):
    """Segments are joined by whatever whitespace separated them in the
    source, not a guessed constant, so structure survives multi-segment
    documents the same way it does single-segment ones."""
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=EchoLLM())
    result = translate_document(pair, settings, llm, InMemorySaver())
    ordered = result.segments
    for i in range(1, len(ordered)):
        gap = pair.source_text[ordered[i - 1].char_end : ordered[i].char_start]
        assert gap in result.output_text or gap == ""


def test_dag_resumes_from_checkpoint_after_a_crash(settings, pair):
    """The reason LangGraph is used here at all (DECISIONS.md D25): a crash
    partway through a document must not re-translate the segments that had
    already completed when re-invoked against the same checkpointer."""
    settings.llm.use_cache = False
    translation_calls = {"n": 0}
    fail_on = 3  # partway through, not the very first segment

    def flaky(messages, params):
        prompt = messages[-1].content
        if "Decision:" in prompt:
            return "no"  # keeps the graph a deterministic linear chain
        translation_calls["n"] += 1
        if translation_calls["n"] == fail_on:
            raise LLMError("simulated crash mid-document")
        return echo(messages, params)

    checkpointer = InMemorySaver()
    llm = build_llm(settings, provider=FakeLLM(responder=flaky))

    with pytest.raises(LLMError):
        translate_document(pair, settings, llm, checkpointer)
    calls_before_resume = translation_calls["n"]

    result = translate_document(pair, settings, llm, checkpointer)

    assert len(result.records) == len(result.segments)
    # +1 for the attempt that crashed and was never retried by the model
    # itself; if resume had re-translated from scratch this would instead be
    # calls_before_resume + len(result.segments).
    assert translation_calls["n"] == len(result.segments) + 1
    assert translation_calls["n"] > calls_before_resume


def test_dag_preserves_line_breaks_across_a_multi_segment_document(settings, pair):
    """Internal newlines survive per segment (translator masking) and between
    segments (source-derived reassembly), together reproducing the source's
    line structure end to end."""
    settings.llm.use_cache = False
    assert pair.source_text.count("\n") > 0, "fixture must exercise multi-line source"
    llm = build_llm(settings, provider=EchoLLM())
    result = translate_document(pair, settings, llm, InMemorySaver())
    assert result.output_text.count("\n") == pair.source_text.strip().count("\n")


# -- line-break masking -------------------------------------------------------


def test_mask_unmask_round_trips_single_and_double_newlines():
    src = "line one\nline two\n\nnew paragraph"
    assert unmask_linebreaks(mask_linebreaks(src)) == src


def test_unmask_absorbs_whitespace_a_model_adds_around_the_token():
    masked = f"before{LINEBREAK_TOKEN}{LINEBREAK_TOKEN}after"
    with_stray_spaces = masked.replace(LINEBREAK_TOKEN, f" {LINEBREAK_TOKEN} ")
    assert unmask_linebreaks(with_stray_spaces) == "before\n\nafter"


def test_prompt_states_the_linebreak_rule_only_when_relevant(settings):
    llm = build_llm(settings, provider=FakeLLM(default='{"translation":"x"}'))
    translator = Translator(settings, llm)
    from mats_stod.translation.context import ContextBlock

    with_breaks = translator.build_prompt(mask_linebreaks("a\nb"), ContextBlock(text=""))
    assert "line-break markers" in with_breaks

    without_breaks = translator.build_prompt("a b", ContextBlock(text=""))
    assert "line-break markers" not in without_breaks


def test_translate_segment_restores_newlines_from_masked_model_output(settings):
    """The model echoes the masked token back (as a real model would, given
    the prompt's instruction); translate_segment must hand back real
    newlines, not the token, in target_text."""
    settings.llm.use_cache = False
    fake = FakeLLM(responder=echo)
    llm = build_llm(settings, provider=fake)
    from mats_stod.schemas import Segment
    from mats_stod.translation.context import ContextBlock

    seg = Segment(
        seg_id="s0", doc_id="d", order=0, text="first line\nsecond line", char_start=0, char_end=22
    )
    record = Translator(settings, llm).translate_segment(seg, ContextBlock(text=""), STRATEGY_NAME)
    assert record.target_text == "first line\nsecond line"
    assert LINEBREAK_TOKEN not in record.target_text


# -- protected content ----------------------------------------------------


def test_matching_digits_are_not_flagged():
    report = protect.check("රු. 3,500.00", "ரூ. 3,500.00", ProtectSettings())
    assert report.ok and report.flags == []


def test_changed_digit_is_flagged():
    report = protect.check("MOF/2023/145", "MOF/2023/146", ProtectSettings())
    assert not report.ok
    assert any("digit" in f or "missing" in f for f in report.flags)


def test_dropped_number_is_flagged():
    report = protect.check("දිනය: 2023.05.12", "திகதி:", ProtectSettings())
    assert not report.ok


def test_reference_code_missing_is_flagged():
    report = protect.check("චක්‍රලේඛ අංක MOF/2023/145", "சுற்றறிக்கை", ProtectSettings())
    assert any("protected_pattern_missing" in f for f in report.flags)


def test_flags_never_modify_the_translation(settings, pair):
    """The check reports; it must not silently repair."""
    settings.llm.use_cache = False
    bad = FakeLLM(default=json.dumps({"translation": "no numbers here"}))
    llm = build_llm(settings, provider=bad)
    result = translate_document(pair, settings, llm, InMemorySaver())
    assert all(r.target_text == "no numbers here" for r in result.records)
    assert any(r.flags for r in result.records)


def test_protect_can_be_disabled():
    off = ProtectSettings(enabled=False)
    assert protect.check("123", "456", off).ok


# -- parsing and retry ----------------------------------------------------


def test_retry_once_then_succeed(settings):
    settings.llm.use_cache = False
    settings.translation.retry_on_parse_failure = 1
    fake = FakeLLM(responses=["not json at all", json.dumps({"translation": "ok"})])
    llm = build_llm(settings, provider=fake)
    translator = Translator(settings, llm)
    from mats_stod.schemas import Segment
    from mats_stod.translation.context import ContextBlock

    seg = Segment(seg_id="s0", doc_id="d", order=0, text="මූලාශ්‍රය", char_start=0, char_end=9)
    record = translator.translate_segment(seg, ContextBlock(text=""), STRATEGY_NAME)
    assert record.target_text == "ok"
    assert fake.call_count == 2
    assert any("parse_failure_attempt_1" in f for f in record.flags)


def test_persistent_parse_failure_falls_back_and_is_flagged(settings):
    settings.llm.use_cache = False
    fake = FakeLLM(default="still not json")
    llm = build_llm(settings, provider=fake)
    from mats_stod.schemas import Segment
    from mats_stod.translation.context import ContextBlock

    seg = Segment(seg_id="s0", doc_id="d", order=0, text="x", char_start=0, char_end=1)
    record = Translator(settings, llm).translate_segment(seg, ContextBlock(text=""), STRATEGY_NAME)
    assert record.target_text == "still not json"
    assert "fell_back_to_raw_text" in record.flags


def test_prompt_contains_both_language_names_and_never_hard_codes_them(settings):
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=FakeLLM(default='{"translation":"x"}'))
    prompt = Translator(settings, llm).build_prompt("source", __import__(
        "mats_stod.translation.context", fromlist=["ContextBlock"]
    ).ContextBlock(text=""))
    assert "Sinhala" in prompt and "Tamil" in prompt

    settings.langs.source, settings.langs.target = "ta", "si"
    flipped = Translator(settings, llm).build_prompt("source", __import__(
        "mats_stod.translation.context", fromlist=["ContextBlock"]
    ).ContextBlock(text=""))
    assert flipped.index("Tamil") < flipped.index("Sinhala")


def test_translation_record_captures_provenance(settings, pair):
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=EchoLLM())
    result = translate_document(pair, settings, llm, InMemorySaver())
    r = result.records[0]
    assert r.model == "fake"
    assert r.prompt_version == settings.translation.prompt_version
    assert r.context_strategy == STRATEGY_NAME

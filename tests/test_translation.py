"""Context strategies, protected content, retry policy."""

from __future__ import annotations

import json

import pytest

from mats_stod.config import ProtectSettings
from mats_stod.io.parallel import load_parallel
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import FakeLLM
from mats_stod.pipelines.baseline import translate_document
from mats_stod.translation import protect
from mats_stod.translation.context import (
    available_strategies,
    build_strategy,
)
from mats_stod.translation.translator import Translator


def echo(messages, params):
    prompt = messages[-1].content
    body = prompt.split("Source text", 1)[1].split(":", 1)[1]
    return json.dumps({"translation": body.split("Return a JSON", 1)[0].strip()}, ensure_ascii=False)


@pytest.fixture
def pair():
    return load_parallel("data/samples", "si", "ta", doc_ids=["circular_01"])[0]


# -- strategies -----------------------------------------------------------


def test_available_strategies():
    assert set(available_strategies()) == {"B0_full_document", "B1_isolated", "B2_sliding_window"}


@pytest.mark.parametrize("short,expected", [("B0", "B0_full_document"), ("B1", "B1_isolated"), ("B2", "B2_sliding_window")])
def test_short_names_resolve(short, expected, settings):
    assert build_strategy(short, settings).name == expected


def test_unknown_strategy_is_rejected(settings):
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("D9", settings)


def test_b0_is_one_call_for_the_whole_document(settings, pair):
    settings.llm.use_cache = False
    settings.translation.strategy = "B0_full_document"
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    assert len(result.segments) == 1
    assert llm.ledger.n_calls == 1


def test_b0_falls_back_to_chunks_and_records_it(settings, pair):
    settings.llm.use_cache = False
    settings.translation.strategy = "B0_full_document"
    settings.translation.full_doc_token_limit = 10  # force the fallback
    settings.translation.chunk_token_size = 40
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    assert len(result.segments) > 1
    assert any("B0 fallback" in n for n in result.notes), "the fallback must be recorded"


def test_b1_sends_no_context(settings, pair):
    settings.llm.use_cache = False
    settings.translation.strategy = "B1_isolated"
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    assert all(r.context_seg_ids == [] for r in result.records)


def test_b2_window_grows_to_k_then_holds(settings, pair):
    settings.llm.use_cache = False
    settings.translation.strategy = "B2_sliding_window"
    settings.translation.window_k = 3
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    sizes = [len(r.context_seg_ids) for r in result.records]
    assert sizes[:4] == [0, 1, 2, 3]
    assert max(sizes) == 3


def test_b2_context_includes_earlier_translations(settings, pair):
    settings.llm.use_cache = False
    settings.translation.strategy = "B2_sliding_window"
    fake = FakeLLM(responder=echo)
    llm = build_llm(settings, provider=fake)
    translate_document(pair, settings, llm)
    assert any("translation:" in p for p in fake.prompts())


def test_context_budget_truncates_oldest_first(settings, pair):
    settings.llm.use_cache = False
    settings.translation.strategy = "B2_sliding_window"
    settings.translation.context_token_budget = 5  # tiny budget
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    assert any(r.metadata.get("context_truncated") for r in result.records)


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
    result = translate_document(pair, settings, llm)
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
    record = translator.translate_segment(seg, ContextBlock(text=""), "B1_isolated")
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
    record = Translator(settings, llm).translate_segment(seg, ContextBlock(text=""), "B1_isolated")
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
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    r = result.records[0]
    assert r.model == "fake"
    assert r.prompt_version == settings.translation.prompt_version
    assert r.context_strategy == settings.translation.strategy

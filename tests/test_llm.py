"""Cache, cost ledger, dry run and the fake provider."""

from __future__ import annotations

import pytest

from mats_stod.llm.base import Message, ParseError, parse_json_response, parse_yes_no
from mats_stod.llm.cache import LLMCache, cache_key
from mats_stod.llm.client import CachedLLM, DryRunExhausted
from mats_stod.llm.cost import CostLedger
from mats_stod.llm.fake import FakeLLM


def msg(text: str) -> list[Message]:
    return [Message("user", text)]


def test_cache_key_is_stable_and_order_independent():
    a = cache_key("p", "m", msg("hello"), None, {"temperature": 0.0, "max_output_tokens": 4})
    b = cache_key("p", "m", msg("hello"), None, {"max_output_tokens": 4, "temperature": 0.0})
    assert a == b


def test_cache_key_changes_with_every_input():
    base = cache_key("p", "m", msg("hello"), None, {"t": 0})
    assert base != cache_key("q", "m", msg("hello"), None, {"t": 0})
    assert base != cache_key("p", "n", msg("hello"), None, {"t": 0})
    assert base != cache_key("p", "m", msg("hello!"), None, {"t": 0})
    assert base != cache_key("p", "m", msg("hello"), {"type": "object"}, {"t": 0})
    assert base != cache_key("p", "m", msg("hello"), None, {"t": 1})


def test_second_identical_call_is_a_cache_hit_and_costs_nothing(tmp_path):
    fake = FakeLLM(default="hi", tokens_in=100, tokens_out=50)
    ledger = CostLedger(prices_usd_per_mtok={"fake": {"input": 1.0, "output": 2.0}})
    llm = CachedLLM(fake, LLMCache(tmp_path / "c.sqlite"), ledger)

    first = llm.complete(msg("q"), purpose="translation")
    second = llm.complete(msg("q"), purpose="translation")

    assert first.cached is False and second.cached is True
    assert fake.call_count == 1, "the provider must be called only once"
    assert ledger.tokens_in_billed == 100
    assert ledger.cost_usd_billed == pytest.approx(100 / 1e6 * 1.0 + 50 / 1e6 * 2.0)
    assert ledger.cost_usd_cold > ledger.cost_usd_billed


def test_dry_run_refuses_an_uncached_call(tmp_path):
    llm = CachedLLM(FakeLLM(default="hi"), LLMCache(tmp_path / "c.sqlite"), CostLedger(), dry_run=True)
    with pytest.raises(DryRunExhausted):
        llm.complete(msg("q"))


def test_dry_run_serves_a_cached_call(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    warm = CachedLLM(FakeLLM(default="hi"), cache, CostLedger())
    warm.complete(msg("q"))
    dry = CachedLLM(FakeLLM(default="other"), cache, CostLedger(), dry_run=True)
    assert dry.complete(msg("q")).text == "hi"


def test_fake_llm_raises_on_an_unscripted_prompt():
    with pytest.raises(Exception, match="no scripted response"):
        FakeLLM().complete(msg("anything"))


def test_fake_llm_rules_match_on_prompt_content():
    fake = FakeLLM(rules=[("Decision", "yes"), ("Source text", '{"translation": "x"}')])
    assert fake.complete(msg("... Decision:")).text == "yes"
    assert fake.complete(msg("... Source text: ...")).text == '{"translation": "x"}'


@pytest.mark.parametrize("raw,expected", [("yes", True), ("Yes", True), ("no", False), ("NO", False), ("y", True)])
def test_parse_yes_no(raw, expected):
    assert parse_yes_no(raw) is expected


@pytest.mark.parametrize("raw", ["", "maybe", "perhaps not"])
def test_parse_yes_no_rejects_other_answers(raw):
    with pytest.raises(ParseError):
        parse_yes_no(raw)


def test_parse_json_tolerates_code_fences():
    assert parse_json_response('```json\n{"translation": "ok"}\n```', key="translation") == "ok"


def test_parse_json_rejects_missing_key():
    with pytest.raises(ParseError):
        parse_json_response('{"other": 1}', key="translation")


def test_ledger_separates_billed_from_cold():
    ledger = CostLedger(prices_usd_per_mtok={"m": {"input": 1.0, "output": 1.0}})
    ledger.record("translation", "m", 1_000_000, 0, cached=False)
    ledger.record("translation", "m", 1_000_000, 0, cached=True)
    assert ledger.cost_usd_billed == pytest.approx(1.0)
    assert ledger.cost_usd_cold == pytest.approx(2.0)
    assert ledger.by_purpose()["translation"]["calls"] == 2


def test_unpriced_model_is_reported_not_treated_as_free():
    """A model missing from the price table must not print $0.00 silently."""
    ledger = CostLedger(prices_usd_per_mtok={"known": {"input": 1.0, "output": 1.0}})
    ledger.record("translation", "brand-new-model", 1_000_000, 0, cached=False)
    summary = ledger.summary()
    assert summary["models_without_prices"] == ["brand-new-model"]
    assert summary["cost_is_complete"] is False


def test_fully_priced_run_is_marked_complete():
    ledger = CostLedger(prices_usd_per_mtok={"known": {"input": 1.0, "output": 1.0}})
    ledger.record("translation", "known", 1_000_000, 0, cached=False)
    summary = ledger.summary()
    assert summary["models_without_prices"] == []
    assert summary["cost_is_complete"] is True
    assert summary["cost_usd_billed"] == pytest.approx(1.0)

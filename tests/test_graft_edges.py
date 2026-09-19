"""GRAFT's pairwise edge agent.

The claim being defended here is a cost claim as much as a quality one: the
method is quadratic, so the call count, the budget ceiling and the behaviour
on an unparseable answer all matter as much as the edges produced.
"""

from __future__ import annotations

import pytest

from mats_stod.graph.graft_edges import (
    GraftPairwiseEdgeInferrer,
    PairwiseCallBudgetError,
    plan_pairs,
)
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import FakeLLM
from mats_stod.schemas import Segment


def segments(n: int) -> list[Segment]:
    return [
        Segment(
            seg_id=f"d-s{i:04d}",
            doc_id="d",
            order=i,
            text=f"discourse number {i}",
            char_start=i * 30,
            char_end=i * 30 + 20,
        )
        for i in range(n)
    ]


def run(settings, answer: str, n: int):
    """Run the inferrer over `n` segments with every answer scripted to `answer`."""
    settings.llm.use_cache = False
    fake = FakeLLM(default=answer)
    llm = build_llm(settings, provider=fake)
    result = GraftPairwiseEdgeInferrer(settings, llm).infer(segments(n))
    return result, fake, llm


# -- the cost claim --------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 11, 20])
def test_pairwise_call_count_matches_the_published_formula(settings, n: int) -> None:
    """(n-1)(n-2)/2 calls. This is the number the whole budget argument rests on."""
    result, fake, _llm = run(settings, "no", n)
    expected = (n - 1) * (n - 2) // 2
    assert len(plan_pairs(n)) == expected
    assert result.stats["n_planned_calls"] == expected
    assert result.stats["llm_calls"] == expected == fake.call_count


def test_adjacent_pairs_are_never_asked_about(settings) -> None:
    """GRAFT links every segment to its successor unconditionally.

    Asking the model about an adjacent pair would pay for an answer that is
    already decided, so those pairs are excluded from the plan.
    """
    assert all(j - i >= 2 for i, j in plan_pairs(6))


def test_a_single_segment_costs_nothing(settings) -> None:
    result, fake, _llm = run(settings, "yes", 1)
    assert result.edges == []
    assert fake.call_count == 0


# -- the budget guard ------------------------------------------------------


def test_budget_ceiling_refuses_before_spending_anything(settings) -> None:
    """The point of a ceiling is to prevent the spend, not to report it.

    A run that discovers the ceiling halfway through has already paid for
    every call it made, so the check must happen before the first one.
    """
    settings.graph.max_pairwise_calls_per_doc = 5
    settings.llm.use_cache = False
    fake = FakeLLM(default="yes")
    llm = build_llm(settings, provider=fake)

    with pytest.raises(PairwiseCallBudgetError) as excinfo:
        GraftPairwiseEdgeInferrer(settings, llm).infer(segments(20))

    assert fake.call_count == 0, "the ceiling must be checked before any call is made"
    assert "171" in str(excinfo.value)  # the number of calls it would have needed
    assert "max_pair_distance" in str(excinfo.value)


def test_max_pair_distance_narrows_the_plan_and_is_a_departure(settings) -> None:
    """The distance bound is an ablation knob; null keeps GRAFT faithful."""
    assert len(plan_pairs(10, max_distance=None)) == 36
    assert len(plan_pairs(10, max_distance=3)) == 15
    settings.graph.max_pair_distance = 3
    result, fake, _llm = run(settings, "no", 10)
    assert fake.call_count == 15
    assert result.stats["n_planned_calls"] == 15


# -- the edges produced ----------------------------------------------------


def test_predecessor_edges_are_unconditional(settings) -> None:
    """Every segment links to its successor whatever the model answers."""
    for answer in ("yes", "no"):
        result, _fake, _llm = run(settings, answer, 6)
        pred = [e for e in result.edges if e.type == "continuation"]
        assert len(pred) == 5 == result.stats["n_predecessor_edges"]
        assert all(e.origin == "structural" for e in pred)


def test_no_answers_leave_only_predecessor_edges(settings) -> None:
    result, _fake, _llm = run(settings, "no", 6)
    assert {e.type for e in result.edges} == {"continuation"}
    assert result.stats["decisions_yes"] == 0
    assert result.stats["yes_rate"] == 0.0


def test_yes_answers_add_untyped_llm_edges(settings) -> None:
    """GRAFT answers one word, so its edges carry no type and no evidence (D6)."""
    result, _fake, _llm = run(settings, "yes", 5)
    llm_edges = [e for e in result.edges if e.origin == "llm"]
    assert len(llm_edges) == (5 - 1) * (5 - 2) // 2 == 6
    assert all(e.type == "graft_dependency" for e in llm_edges)
    assert all(e.evidence is None and e.confidence is None for e in llm_edges)
    assert result.stats["yes_rate"] == 1.0


def test_edges_always_point_forward_in_reading_order(settings) -> None:
    result, _fake, _llm = run(settings, "yes", 8)
    for e in result.edges:
        assert int(e.src[-4:]) < int(e.dst[-4:])


def test_unparseable_answer_counts_as_not_connected(settings) -> None:
    """D18: the conservative reading.

    A missing edge weakens the context; an invented one corrupts it. An
    unparseable answer must never become an edge, and must be counted so the
    rate is visible in the report.
    """
    result, _fake, _llm = run(settings, "perhaps, it depends", 5)
    assert [e for e in result.edges if e.origin == "llm"] == []
    assert result.stats["decisions_unparsed"] == 6
    assert result.stats["decisions_yes"] == 0


def test_the_model_is_asked_for_one_token_with_thinking_disabled(settings) -> None:
    """A thinking model would spend the budget reasoning and return nothing.

    That failure is silent: every pair reads as "not connected" and the run
    still looks successful, so the parameters are asserted rather than trusted.
    """
    _r, fake, llm = run(settings, "yes", 4)
    params = fake.calls[0]["params"]
    assert params["max_output_tokens"] == settings.graph.decision_max_tokens
    assert params["thinking_budget"] == 0
    assert params["temperature"] == 0.0
    # `purpose` is consumed by CachedLLM for the cost ledger, so it is
    # asserted where it lands: the report groups spend by this label.
    assert "edge_inference" in llm.ledger.by_purpose()


def test_the_prompt_carries_both_discourses_and_the_language_names(settings) -> None:
    _r, fake, _llm = run(settings, "no", 3)
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "discourse number 0" in prompt
    assert "discourse number 2" in prompt
    assert settings.langs.source_name in prompt


# -- the density finding ---------------------------------------------------


def test_yes_rate_is_reported_because_a_dense_graph_collapses_d1(settings) -> None:
    """The share of pairs answered yes decides whether D1 can differ from B0.

    Reported on every run so the finding arrives before M4 is designed.
    """
    result, _fake, _llm = run(settings, "yes", 6)
    assert result.stats["yes_rate"] == 1.0
    assert result.stats["n_planned_calls"] == 10

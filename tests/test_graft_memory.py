"""Behavioral tests for the paper-aligned memory baseline, entirely offline."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from mats_stod.config import load_settings
from mats_stod.io.parallel import load_parallel
from mats_stod.llm.base import LLMError
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import EchoLLM, FakeLLM
from mats_stod.pipelines import dag_translate
from mats_stod.pipelines.graph_pipeline import DocumentGraph, build_document_graph
from mats_stod.schemas import DiscourseGraph, Edge, Segment, TranslationRecord
from mats_stod.translation.context import ContextBlock
from mats_stod.translation.memory import (
    DiscourseMemory,
    MemoryAgent,
    MemoryExtractionError,
    MemoryRecord,
    RequestCapacityError,
    merge_memories,
)
from mats_stod.translation.translator import Translator


def payload(summary="Local summary", **components):
    return {
        "noun_pronoun": [],
        "entities": [],
        "phrases": [],
        "connectives": [],
        "summary": summary,
        **components,
    }


def graph(n=4, dense=False):
    segments = [
        Segment(
            seg_id=f"s{i}",
            doc_id="d",
            order=i,
            text=f"Text {i}.",
            char_start=i * 8,
            char_end=i * 8 + 7,
        )
        for i in range(n)
    ]
    edges = [
        Edge(src=f"s{i}", dst=f"s{j}", type="continuation")
        for j in range(1, n)
        for i in (range(j) if dense else [j - 1])
    ]
    return DiscourseGraph(doc_id="d", segments=segments, edges=edges)


@pytest.fixture
def baseline(settings):
    settings.translation.condition = "graft_baseline"
    settings.segmentation.max_discourse_chars = None
    settings.llm.use_cache = False
    return settings


def test_real_defaults_are_unpruned_memory_baseline():
    settings = load_settings()
    assert settings.translation.condition == "graft_baseline"
    assert settings.graph.max_parents == 0
    assert settings.graph.max_pair_distance is None
    assert settings.graph.transitive_reduction is False
    assert settings.segmentation.max_discourse_chars is None


@pytest.mark.parametrize("dense", [False, True])
def test_alternating_stages_and_direct_parents_only(baseline, dense):
    g = graph(dense=dense)
    llm = build_llm(baseline, provider=EchoLLM())
    app = dag_translate._build_memory_app(g, baseline, llm).compile()
    events = list(
        app.stream({"records": {}, "memories": {}, "memory_contexts": {}}, stream_mode="updates")
    )
    assert [next(iter(e)) for e in events] == [
        name for i in range(4) for name in (f"translate_{i}", f"memory_{i}")
    ]
    assert [c.purpose for c in llm.ledger.calls] == ["translation", "memory_extraction"] * 4
    for i in range(4):
        update = events[2 * i][f"translate_{i}"]
        context = update["memory_contexts"][f"s{i}"]
        assert context["parent_ids"] == g.parents(f"s{i}")
        assert "Text 0." not in context["prompt_context"]  # memory, not whole sources


def test_earliest_exact_key_wins_and_conflicts_keep_provenance():
    g = graph(3)
    segments = {s.seg_id: s for s in g.segments}
    memories = {
        sid: MemoryRecord(
            seg_id=sid,
            model="fake",
            prompt_version="v1",
            memory=DiscourseMemory.model_validate(
                payload(
                    summary=f"Summary {sid}", entities=[{"source": "Authority", "target": target}]
                )
            ),
        )
        for sid, target in [("s0", "Department"), ("s1", "Board")]
    }
    context, audit = merge_memories(["s1", "s0", "s0"], segments, memories)
    assert context.seg_ids == ["s0", "s1"]
    assert audit["memory"]["entities"] == [
        {"source": "Authority", "target": "Department", "seg_id": "s0"}
    ]
    assert audit["conflicts"][0]["discarded"]["seg_id"] == "s1"
    assert len(audit["memory"]["summaries"]) == 2
    with pytest.raises(MemoryExtractionError, match="Missing parent"):
        merge_memories(["s2"], segments, memories)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        payload(summary=""),
        payload(entities=[{"source": "x"}]),
        payload(extra="unknown"),
        payload(summary=123),
    ],
)
def test_invalid_memory_is_rejected(bad):
    with pytest.raises(ValidationError):
        DiscourseMemory.model_validate(bad)


def test_extraction_retries_and_accounts_for_both_calls(baseline):
    llm = build_llm(baseline, provider=FakeLLM(responses=["bad", json.dumps(payload())]))
    seg = graph(1).segments[0]
    trans = TranslationRecord(
        seg_id=seg.seg_id,
        source_text=seg.text,
        target_text="Done",
        context_strategy="graft_baseline",
        model="fake",
        prompt_version="v1",
    )
    result = MemoryAgent(baseline, llm).extract(seg, trans)
    assert len(result.parse_failures) == 1 and result.tokens_in == 20
    assert len(llm.ledger.provider_purposes) == 2
    prompts = [call["messages"][0]["content"] for call in llm.provider.calls]
    assert prompts[0] != prompts[1]
    assert "Done" in prompts[0] and seg.text in prompts[0]
    assert all(call["params"]["thinking_budget"] == 0 for call in llm.provider.calls)


def test_memory_schema_and_prompt_bound_extraction():
    from mats_stod.prompts.registry import render
    from mats_stod.translation.memory import MEMORY_SCHEMA

    assert MEMORY_SCHEMA["properties"]["entities"]["maxItems"] == 10
    prompt = render(
        "memory_v1", source_lang="Sinhala", target_lang="Tamil", source="Text.", translation="உரை."
    )
    assert "at most 10 salient" in prompt


def test_failed_memory_resumes_saved_translation_with_sqlite(baseline, tmp_path):
    baseline.llm.use_cache = True
    g = graph(2)
    failed = False
    echo = EchoLLM()

    def responder(messages, params):
        nonlocal failed
        if messages[0].content.startswith("GRAFT local memory") and not failed:
            failed = True
            return "{}"
        return echo.complete(messages).text

    baseline.memory.retry_on_parse_failure = 0
    llm = build_llm(baseline, provider=FakeLLM(responder=responder))
    config = {"configurable": {"thread_id": "resume"}}
    path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        app = dag_translate._build_memory_app(g, baseline, llm).compile(checkpointer=saver)
        with pytest.raises(MemoryExtractionError):
            app.invoke({"records": {}, "memories": {}, "memory_contexts": {}}, config)
        saved = saver.get_tuple(config).checkpoint["channel_values"]
        assert list(saved["records"]) == ["s0"] and saved["memories"] == {}
    with SqliteSaver.from_conn_string(path) as saver:
        app = dag_translate._build_memory_app(g, baseline, llm).compile(checkpointer=saver)
        final = app.invoke(None, config)
    assert len(final["memories"]) == 2
    assert llm.ledger.by_purpose()["translation"]["calls"] == 2


def test_full_request_capacity_stops_without_translation_call(baseline):
    baseline.memory.request_token_limit = 10
    llm = build_llm(baseline, provider=EchoLLM())
    with pytest.raises(RequestCapacityError, match="No context was truncated"):
        Translator(baseline, llm).translate_segment(
            graph(1).segments[0], ContextBlock(text="Large memory"), "graft_baseline"
        )
    assert llm.ledger.n_calls == 0


def test_baseline_does_not_fall_back_to_unparsed_translation(baseline):
    llm = build_llm(baseline, provider=FakeLLM(default="bad json"))
    with pytest.raises(LLMError, match="refusing raw fallback"):
        Translator(baseline, llm).translate_segment(
            graph(1).segments[0], ContextBlock(text=""), "graft_baseline"
        )


def test_graph_keeps_all_positive_edges_and_rejects_pruning(baseline):
    pair = load_parallel("data/samples", "si", "ta", max_docs=1)[0]

    def decisions(messages, params):
        # Keep sentence segments, approve every inferred edge.
        return "yes" if "Discourse 1:" in messages[0].content else "no"

    llm = build_llm(baseline, provider=FakeLLM(responder=decisions))
    doc = build_document_graph(pair, baseline, llm)
    n = len(doc.graph.segments)
    assert n > 5 and len(doc.graph.edges) == n * (n - 1) // 2
    assert len(doc.graph.parents(doc.graph.segments[-1].seg_id)) == n - 1
    baseline.graph.max_parents = 4
    before = llm.ledger.n_calls
    with pytest.raises(ValueError, match="requires max_parents"):
        build_document_graph(pair, baseline, llm)
    assert llm.ledger.n_calls == before


def test_baseline_ignores_raw_depth_budget_and_reference(baseline, monkeypatch):
    g = graph(3)
    pair = load_parallel("data/samples", "si", "ta", max_docs=1)[0]
    pair.source_text = "Text 0. Text 1. Text 2."
    pair.reference_text = "SECRET REFERENCE MUST NOT ENTER PROMPTS"
    monkeypatch.setattr(
        dag_translate,
        "build_document_graph",
        lambda *args: DocumentGraph("d", pair.direction, "si", g),
    )
    baseline.translation.dag_context_depth = 0
    baseline.translation.context_token_budget = 0
    llm = build_llm(baseline, provider=EchoLLM())
    result = dag_translate.translate_document(pair, baseline, llm, InMemorySaver())
    assert len(result.memories) == 3
    assert result.records[2].context_seg_ids == ["s1"]
    assert all("SECRET REFERENCE" not in c["messages"][0]["content"] for c in llm.provider.calls)

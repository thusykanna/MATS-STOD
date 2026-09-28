"""Scheduling, context, and checkpoint regressions for dense discourse graphs."""

from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from mats_stod.graph.assemble import assemble_graph
from mats_stod.io.parallel import DocPair
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import EchoLLM
from mats_stod.pipelines import dag_translate
from mats_stod.pipelines.graph_pipeline import DocumentGraph
from mats_stod.schemas import DiscourseGraph, Edge, Segment, TranslationRecord
from mats_stod.translation.context import build_dag_context


def make_graph(n, pairs):
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
    return DiscourseGraph(
        doc_id="d",
        segments=segments,
        edges=[Edge(src=f"s{i}", dst=f"s{j}", type="continuation") for i, j in pairs],
    )


class RecordingTranslator:
    def __init__(self):
        self.calls = []
        self.fail_once = None

    def translate_segment(self, segment, context, strategy):
        if segment.seg_id == self.fail_once:
            self.fail_once = None
            raise RuntimeError("interrupted")
        # Every selected ancestor must have a completed translation in the prompt.
        assert context.text.count("translation:") == len(context.seg_ids)
        self.calls.append((segment.seg_id, context))
        return TranslationRecord(
            seg_id=segment.seg_id,
            source_text=segment.text,
            target_text=f"Translated {segment.seg_id}",
            context_strategy=strategy,
            model="demo",
            prompt_version="demo",
            context_seg_ids=context.seg_ids,
        )


def app_for(graph, settings, translator, saver=None):
    return dag_translate._build_graph_app(
        graph, {s.seg_id: s for s in graph.segments}, settings, translator
    ).compile(checkpointer=saver)


@pytest.mark.parametrize(
    "n,pairs",
    [
        (1, []),
        (4, [(0, 1), (1, 2), (2, 3)]),
        (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        (5, [(0, 1), (0, 2), (2, 3), (1, 4), (3, 4)]),  # unequal branch lengths
        (8, [(i, j) for j in range(8) for i in range(j)]),
    ],
)
def test_each_node_executes_once_after_dependencies(settings, n, pairs):
    graph = make_graph(n, pairs)
    translator = RecordingTranslator()
    state = app_for(graph, settings, translator).invoke({"records": {}})
    assert Counter(sid for sid, _ in translator.calls) == Counter(s.seg_id for s in graph.segments)
    assert len(state["records"]) == n
    finished = set()
    for sid, context in translator.calls:
        assert set(graph.parents(sid)) <= finished
        assert set(context.seg_ids) <= finished
        finished.add(sid)


@pytest.mark.parametrize("depth,expected", [(0, []), (1, ["s1"]), (2, ["s0", "s1"])])
def test_context_depth(settings, depth, expected):
    settings.translation.dag_context_depth = depth
    translator = RecordingTranslator()
    app_for(make_graph(3, [(0, 1), (1, 2)]), settings, translator).invoke({"records": {}})
    assert dict(translator.calls)["s2"].seg_ids == expected


def test_missing_dependency_is_an_error(settings):
    graph = make_graph(2, [(0, 1)])
    node = dag_translate._make_node(
        "s1", ["s0"], [], {s.seg_id: s for s in graph.segments}, settings, RecordingTranslator()
    )
    with pytest.raises(RuntimeError, match="missing dependency records"):
        node({"records": {}})


def test_dense_graph_resume_does_not_repeat_completed_nodes(settings):
    graph = make_graph(4, [(i, j) for j in range(4) for i in range(j)])
    translator = RecordingTranslator()
    translator.fail_once = "s2"
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "test"}}
    with pytest.raises(RuntimeError, match="interrupted"):
        app_for(graph, settings, translator, saver).invoke({"records": {}}, config)
    assert [sid for sid, _ in translator.calls] == ["s0", "s1"]
    final = app_for(graph, settings, translator, saver).invoke(None, config)
    assert len(final["records"]) == 4
    assert [sid for sid, _ in translator.calls] == ["s0", "s1", "s2", "s3"]


def test_context_budget_diagnostics(settings):
    graph = make_graph(3, [])
    segments = {s.seg_id: s for s in graph.segments}
    settings.translation.chars_per_token_estimate = 1
    settings.translation.context_token_budget = len("[2] source: Text 2.") + 1
    context = build_dag_context(["s2", "s0", "s1", "s0"], segments, {}, settings)
    assert context.selected_seg_ids == ["s0", "s1", "s2"]
    assert context.seg_ids == ["s2"]
    assert context.dropped_seg_ids == ["s0", "s1"]
    assert context.estimated_tokens_before > context.estimated_tokens_after
    settings.translation.context_token_budget = 0
    context = build_dag_context(["s0"], segments, {}, settings)
    assert context.text == "" and context.seg_ids == []
    assert context.empty_reason == "context_budget_exhausted"
    assert context.dropped_seg_ids == ["s0"]


def test_parent_cap_and_reduction_are_visible(settings):
    graph = make_graph(6, [(i, j) for j in range(6) for i in range(j)])
    settings.graph.max_parents = 4
    capped, stats = assemble_graph("d", graph.segments, graph.edges, settings)
    assert capped.parents("s5") == ["s1", "s2", "s3", "s4"]
    assert stats.as_dict()["dropped_parent_ids"]["s5"] == ["s0"]
    settings.graph.transitive_reduction = True
    reduced, stats = assemble_graph("d", graph.segments, graph.edges, settings)
    assert len(reduced.edges) == 5
    assert reduced.parents("s5") == ["s4"]
    assert reduced.ancestors("s5", 2) == ["s3", "s4"]
    assert stats.dropped_transitive > 0


def test_empty_document(settings):
    pair = DocPair(
        doc_id="empty", source_lang="si", target_lang="ta", source_text="", reference_text=""
    )
    llm = build_llm(settings, provider=EchoLLM())
    result = dag_translate.translate_document(pair, settings, llm, InMemorySaver())
    assert result.output_text == "" and result.records == []
    assert llm.ledger.n_calls == 0


def test_checkpoint_rejects_changes_before_graph_requests(settings, monkeypatch):
    graph = make_graph(1, [])
    pair = DocPair(
        doc_id="d",
        source_lang="si",
        target_lang="ta",
        source_text="Text 0.",
        reference_text="reference",
    )
    monkeypatch.setattr(
        dag_translate,
        "build_document_graph",
        lambda *args: DocumentGraph("d", "si-ta", "si", graph),
    )
    llm = build_llm(settings, provider=EchoLLM())
    saver = InMemorySaver()
    dag_translate.translate_document(pair, settings, llm, saver)
    n_calls = llm.ledger.n_calls
    dag_translate.translate_document(pair, settings, llm, saver)
    assert llm.ledger.n_calls == n_calls
    settings.translation.dag_context_depth += 1
    monkeypatch.setattr(
        dag_translate,
        "build_document_graph",
        lambda *args: pytest.fail("must reject before graph calls"),
    )
    with pytest.raises(dag_translate.CheckpointCompatibilityError, match="fresh --run-id"):
        dag_translate.translate_document(pair, settings, llm, saver)


def test_legacy_checkpoint_is_rejected(settings):
    graph = make_graph(1, [])
    saver = InMemorySaver()
    app_for(graph, settings, RecordingTranslator(), saver).invoke(
        {"records": {}}, {"configurable": {"thread_id": "si-ta:d"}}
    )
    pair = DocPair(
        doc_id="d", source_lang="si", target_lang="ta", source_text="Text 0.", reference_text=""
    )
    with pytest.raises(dag_translate.CheckpointCompatibilityError, match="legacy"):
        dag_translate.translate_document(
            pair, settings, build_llm(settings, provider=EchoLLM()), saver
        )


def test_interrupted_parallel_branch_keeps_successful_sibling(settings):
    graph = make_graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    translator = RecordingTranslator()
    translator.fail_once = "s2"
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "diamond"}}
    with pytest.raises(RuntimeError, match="interrupted"):
        app_for(graph, settings, translator, saver).invoke({"records": {}}, config)
    app_for(graph, settings, translator, saver).invoke(None, config)
    assert Counter(sid for sid, _ in translator.calls) == Counter(["s0", "s1", "s2", "s3"])


def test_rebuilt_graph_change_is_rejected(settings, monkeypatch):
    graph = make_graph(2, [(0, 1)])
    pair = DocPair(
        doc_id="d",
        source_lang="si",
        target_lang="ta",
        source_text="Text 0. Text 1.",
        reference_text="",
    )
    monkeypatch.setattr(
        dag_translate,
        "build_document_graph",
        lambda *args: DocumentGraph("d", "si-ta", "si", graph),
    )
    llm = build_llm(settings, provider=EchoLLM())
    saver = InMemorySaver()
    dag_translate.translate_document(pair, settings, llm, saver)
    n_calls = llm.ledger.n_calls
    graph.edges = []
    with pytest.raises(dag_translate.CheckpointCompatibilityError, match="Rebuilt graph"):
        dag_translate.translate_document(pair, settings, llm, saver)
    assert llm.ledger.n_calls == n_calls


def test_record_persists_empty_context_diagnostics(settings):
    from mats_stod.translation.translator import Translator

    graph = make_graph(2, [])
    settings.translation.context_token_budget = 0
    context = build_dag_context(["s0"], {s.seg_id: s for s in graph.segments}, {}, settings)
    record = Translator(settings, build_llm(settings, provider=EchoLLM())).translate_segment(
        graph.segments[1], context, "test"
    )
    assert record.context_seg_ids == []
    assert record.metadata["context_selected_seg_ids"] == ["s0"]
    assert record.metadata["context_dropped_seg_ids"] == ["s0"]
    assert record.metadata["context_empty_reason"] == "context_budget_exhausted"
    assert record.metadata["context_estimated_tokens_after"] == 0

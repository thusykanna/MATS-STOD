"""DAG-context translation (D1): the discourse graph drives translation order and context.

A document is segmented and its discourse graph built exactly as
`graph_pipeline.py` does. Translation then runs as a LangGraph state graph
whose nodes and edges mirror that discourse graph one-for-one: a segment's
node fires only once every segment it depends on has already been
translated, and receives those translations as its context (`DiscourseGraph
.ancestors`, via `translation.context.build_dag_context`). Segments with no
dependency relationship between them fall into the same superstep and run
concurrently, so the graph structure buys real parallelism, not just an
ordering guarantee.

LangGraph is used here, and nowhere else in the pipeline, per DECISIONS.md
D25: this is the one stage with real per-node state and a use for
checkpointing. Progress is checkpointed to SQLite keyed by (run, doc_id), so
a run interrupted partway resumes from its last completed segment instead of
re-translating the document, when re-invoked against the same run directory
and thread id.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..graph.export import write_graphml, write_json, write_mermaid
from ..io.parallel import DocPair
from ..io.runs import RunDir
from ..llm.client import CachedLLM
from ..schemas import DiscourseGraph, Segment, TranslationRecord
from ..translation.context import build_dag_context
from ..translation.translator import Translator
from .graph_pipeline import build_document_graph

STRATEGY_NAME = "D1_dag_context"


class _TranslationState(TypedDict):
    #: Keyed by seg_id, holding a TranslationRecord dumped to a plain dict so
    #: the checkpointer serialises it without depending on pydantic support.
    #: `operator.or_` merges the partial updates concurrent nodes each return
    #: in the same superstep, rather than one overwriting another's.
    records: Annotated[dict[str, dict[str, Any]], operator.or_]


@dataclass
class DocumentTranslation:
    """Everything one document produced under the DAG-context condition."""

    doc_id: str
    direction: str
    strategy: str
    segments: list[Segment]
    records: list[TranslationRecord]
    output_text: str
    reference_text: str
    source_text: str
    #: The discourse graph translation order and context were drawn from,
    #: kept so it can be exported alongside the translation that used it.
    graph: DiscourseGraph
    graph_stats: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _make_node(
    seg_id: str,
    parent_ids: list[str],
    segments_by_id: dict[str, Segment],
    settings: Settings,
    translator: Translator,
):
    def node(state: _TranslationState) -> dict[str, Any]:
        records = {sid: TranslationRecord.model_validate(r) for sid, r in state["records"].items()}
        context = build_dag_context(parent_ids, segments_by_id, records, settings)
        record = translator.translate_segment(segments_by_id[seg_id], context, STRATEGY_NAME)
        return {"records": {seg_id: record.model_dump(mode="json")}}

    return node


def _build_graph_app(
    graph: DiscourseGraph,
    segments_by_id: dict[str, Segment],
    settings: Settings,
    translator: Translator,
):
    """A LangGraph StateGraph with one node per segment, wired from `graph`'s edges."""
    builder = StateGraph(_TranslationState)
    parents_of = {s.seg_id: graph.parents(s.seg_id) for s in graph.segments}
    children_of = {s.seg_id: graph.children(s.seg_id) for s in graph.segments}

    for seg_id, parent_ids in parents_of.items():
        node = _make_node(seg_id, parent_ids, segments_by_id, settings, translator)
        builder.add_node(seg_id, node)

    for seg_id, parent_ids in parents_of.items():
        if not parent_ids:
            builder.add_edge(START, seg_id)
        for parent_id in parent_ids:
            builder.add_edge(parent_id, seg_id)
        if not children_of[seg_id]:
            builder.add_edge(seg_id, END)

    return builder


def _reassemble(source_text: str, ordered: list[Segment], records: list[TranslationRecord]) -> str:
    """Join translated segments with the exact whitespace that separated their sources.

    A fixed separator (a blank line, a single join character) would be a
    guess; the source text already says what stood between two segments,
    whether that is a paragraph break, a single line break inside a letter
    header, or nothing at all.
    """
    parts: list[str] = []
    for i, (seg, rec) in enumerate(zip(ordered, records, strict=True)):
        if i > 0:
            parts.append(source_text[ordered[i - 1].char_end : seg.char_start])
        parts.append(rec.target_text)
    return "".join(parts)


def translate_document(
    pair: DocPair,
    settings: Settings,
    llm: CachedLLM,
    checkpointer: BaseCheckpointSaver,
) -> DocumentTranslation:
    """Segment, build the discourse graph, and translate it under the DAG-context condition."""
    doc = build_document_graph(pair, settings, llm)
    ordered = sorted(doc.graph.segments, key=lambda s: s.order)

    if not ordered:
        return DocumentTranslation(
            doc_id=pair.doc_id,
            direction=pair.direction,
            strategy=STRATEGY_NAME,
            segments=[],
            records=[],
            output_text="",
            reference_text=pair.reference_text,
            source_text=pair.source_text,
            graph=doc.graph,
            graph_stats=doc.stats,
            stats={"n_flagged_segments": 0},
        )

    segments_by_id = {s.seg_id: s for s in ordered}
    translator = Translator(settings, llm)
    builder = _build_graph_app(doc.graph, segments_by_id, settings, translator)
    app = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {"thread_id": f"{pair.direction}:{pair.doc_id}"},
        # A worst-case discourse graph is one linear chain (GRAFT links every
        # segment to its successor unconditionally), which needs one
        # superstep per segment. The default limit comfortably covers that
        # for any realistic document, but the margin is cheap and removes
        # the doubt.
        "recursion_limit": len(ordered) + 10,
    }
    # Invoking with a fresh input always (re)starts the thread from scratch,
    # discarding whatever the checkpointer already has for it. Resuming means
    # invoking with `None`, which tells LangGraph to continue the existing
    # thread's state instead — the entire reason a checkpointer was worth
    # adding (D25).
    resuming = checkpointer.get_tuple(config) is not None
    final_state = app.invoke(None if resuming else {"records": {}}, config=config)

    records_by_id = {
        seg_id: TranslationRecord.model_validate(rec)
        for seg_id, rec in final_state["records"].items()
    }
    records = [records_by_id[s.seg_id] for s in ordered]
    output = _reassemble(pair.source_text, ordered, records)

    stats = {"n_flagged_segments": sum(1 for r in records if r.flags)}

    return DocumentTranslation(
        doc_id=pair.doc_id,
        direction=pair.direction,
        strategy=STRATEGY_NAME,
        segments=ordered,
        records=records,
        output_text=output,
        reference_text=pair.reference_text,
        source_text=pair.source_text,
        graph=doc.graph,
        graph_stats=doc.stats,
        stats=stats,
    )


def write_document_artifacts(run: RunDir, result: DocumentTranslation) -> None:
    """Persist one document's artifacts so any stage can be re-run alone."""
    base = run.subdir("documents") / result.doc_id
    base.mkdir(parents=True, exist_ok=True)
    run.write_json(f"documents/{result.doc_id}/segments.json", result.segments)
    run.write_json(f"documents/{result.doc_id}/records.json", result.records)
    run.write_text(f"documents/{result.doc_id}/translation.txt", result.output_text + "\n")
    run.write_json(
        f"documents/{result.doc_id}/stats.json",
        {"stats": result.stats, "graph_stats": result.graph_stats, "direction": result.direction},
    )
    if result.segments:
        write_json(result.graph, base / "graph.json")
        write_mermaid(result.graph, base / "graph.mmd")
        write_graphml(result.graph, base / "graph.graphml")


def checkpoint_path(run: RunDir) -> Path:
    """Where this run's LangGraph checkpoints live.

    One file per run, with documents distinguished by thread id, rather than
    one file per document: the checkpointer's own SQLite connection already
    serialises concurrent writers, and one small file is easier to inspect or
    delete than many.
    """
    p = Path(run.path) / "checkpoints.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

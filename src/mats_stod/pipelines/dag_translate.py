"""Terminology-aware GRAFT memory baseline and raw-context comparison.

A document is segmented and connected through graph_pipeline. The default first
resolves terminology for every segment, then executes alternating translation
and local-memory extraction stages in reading order, merging only direct-parent
memories. The retained raw-context condition uses an all-parent barrier per
segment and depth-selected raw passages.

LangGraph is used here, and nowhere else in the pipeline, per DECISIONS.md
D25: this is the one stage with real per-node state and a use for
checkpointing. Progress is checkpointed to SQLite keyed by (run, doc_id), so
a run interrupted partway resumes from its last completed segment instead of
re-translating the document, when re-invoked against the same run directory
and thread id.
"""

from __future__ import annotations

import hashlib
import json
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
from ..prompts.registry import prompt_text
from ..schemas import DiscourseGraph, Segment, TranslationRecord
from ..terminology import GlossaryStore, TerminologyAgent, TerminologyRecord
from ..translation.context import build_dag_context
from ..translation.memory import MemoryAgent, MemoryRecord, merge_memories
from ..translation.translator import Translator
from .graph_pipeline import build_document_graph

STRATEGY_NAME = "dag_raw_context"


def strategy_name(settings: Settings) -> str:
    return "graft_baseline" if settings.translation.condition == "graft_baseline" else STRATEGY_NAME


class _TranslationState(TypedDict):
    fingerprint: str
    graph_fingerprint: str
    #: Keyed by seg_id, holding a TranslationRecord dumped to a plain dict so
    #: the checkpointer serialises it without depending on pydantic support.
    #: `operator.or_` merges the partial updates concurrent nodes each return
    #: in the same superstep, rather than one overwriting another's.
    records: Annotated[dict[str, dict[str, Any]], operator.or_]
    memories: Annotated[dict[str, dict[str, Any]], operator.or_]
    memory_contexts: Annotated[dict[str, dict[str, Any]], operator.or_]
    terminology: Annotated[dict[str, dict[str, Any]], operator.or_]


@dataclass
class DocumentTranslation:
    """Everything one document produced under its configured condition."""

    doc_id: str
    direction: str
    strategy: str
    segments: list[Segment]
    records: list[TranslationRecord]
    output_text: str
    reference_text: str | None
    source_text: str
    #: The discourse graph translation order and context were drawn from,
    #: kept so it can be exported alongside the translation that used it.
    graph: DiscourseGraph
    graph_stats: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    memories: list[MemoryRecord] = field(default_factory=list)
    memory_contexts: dict[str, Any] = field(default_factory=dict)
    terminology: list[TerminologyRecord] = field(default_factory=list)
    glossary_snapshot: dict[str, Any] | None = None


def _make_node(
    seg_id: str,
    parent_ids: list[str],
    context_ids: list[str],
    segments_by_id: dict[str, Segment],
    settings: Settings,
    translator: Translator,
):
    def node(state: _TranslationState) -> dict[str, Any]:
        records = {sid: TranslationRecord.model_validate(r) for sid, r in state["records"].items()}
        missing = sorted(set(parent_ids + context_ids) - records.keys())
        if missing:
            raise RuntimeError(f"Cannot translate {seg_id}: missing dependency records {missing}")
        context = build_dag_context(context_ids, segments_by_id, records, settings)
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
        context_ids = graph.ancestors(seg_id, settings.translation.dag_context_depth)
        node = _make_node(seg_id, parent_ids, context_ids, segments_by_id, settings, translator)
        builder.add_node(seg_id, node)

    for seg_id, parent_ids in parents_of.items():
        if not parent_ids:
            builder.add_edge(START, seg_id)
        else:
            builder.add_edge(parent_ids, seg_id)
        if not children_of[seg_id]:
            builder.add_edge(seg_id, END)

    return builder


def _build_memory_app(
    graph: DiscourseGraph,
    settings: Settings,
    llm: CachedLLM,
    glossary: GlossaryStore | None = None,
):
    """Terminology prepass, then the reading-order translate → memory loop."""
    builder = StateGraph(_TranslationState)
    segments = {s.seg_id: s for s in graph.segments}
    translator = Translator(settings, llm)
    agent = MemoryAgent(settings, llm)
    terminology_agent: TerminologyAgent | None = None
    if settings.terminology.enabled:
        glossary = glossary or GlossaryStore.from_settings(settings)
        glossary.validate_direction(settings.langs.source, settings.langs.target)
        terminology_agent = TerminologyAgent(settings, llm, glossary)

    def terminology_node(sid):
        def run(state):
            assert terminology_agent is not None
            record = terminology_agent.extract(segments[sid])
            return {"terminology": {sid: record.model_dump(mode="json")}}

        return run

    def translation_node(sid):
        def run(state):
            memories = {k: MemoryRecord.model_validate(v)
                        for k, v in state["memories"].items()}
            context, audit = merge_memories(graph.parents(sid), segments, memories)
            terminology = None
            if settings.terminology.enabled:
                raw = state["terminology"].get(sid)
                if raw is None:
                    raise RuntimeError(f"Cannot translate {sid}: terminology prepass is incomplete")
                terminology = TerminologyRecord.model_validate(raw)
            record = translator.translate_segment(
                segments[sid], context, "graft_baseline", terminology
            )
            return {"records": {sid: record.model_dump(mode="json")},
                    "memory_contexts": {sid: audit}}
        return run

    def memory_node(sid):
        def run(state):
            record = TranslationRecord.model_validate(state["records"][sid])
            memory = agent.extract(segments[sid], record)
            return {"memories": {sid: memory.model_dump(mode="json")}}
        return run

    ordered = sorted(graph.segments, key=lambda s: s.order)
    previous: str | list[str] = START
    if terminology_agent is not None:
        terminology_nodes: list[str] = []
        for segment in ordered:
            name = f"terminology_{segment.order}"
            builder.add_node(name, terminology_node(segment.seg_id))
            builder.add_edge(START, name)
            terminology_nodes.append(name)
        # A list-valued edge is an all-node barrier: no translation starts
        # until the terminology record for every segment has checkpointed.
        previous = terminology_nodes

    for segment in ordered:
        sid = segment.seg_id
        trans, mem = f"translate_{segment.order}", f"memory_{segment.order}"
        builder.add_node(trans, translation_node(sid))
        builder.add_node(mem, memory_node(sid))
        builder.add_edge(previous, trans)
        builder.add_edge(trans, mem)
        previous = mem
    builder.add_edge(previous, END)
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


class CheckpointCompatibilityError(RuntimeError):
    """Saved state belongs to a different input or execution contract."""


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def translate_document(
    pair: DocPair,
    settings: Settings,
    llm: CachedLLM,
    checkpointer: BaseCheckpointSaver,
) -> DocumentTranslation:
    """Build the graph and execute the selected translation condition."""
    terminology_active = (
        settings.translation.condition == "graft_baseline" and settings.terminology.enabled
    )
    glossary = GlossaryStore.from_settings(settings) if terminology_active else None
    if glossary is not None:
        glossary.validate_direction(settings.langs.source, settings.langs.target)
    prompt_versions = [
        settings.segmentation.prompt_version,
        settings.graph.prompt_version,
        settings.translation.prompt_version,
        settings.memory.prompt_version,
    ]
    if terminology_active:
        prompt_versions.append(settings.terminology.prompt_version)
    config = {"configurable": {"thread_id": f"{pair.direction}:{pair.doc_id}"}}
    fingerprint = _fingerprint({
        "execution_version": 4,
        "source": pair.source_text,
        "settings": settings.model_dump(mode="json"),
        "prompts": {v: prompt_text(v) for v in prompt_versions},
        "glossary_hash": glossary.content_hash if glossary is not None else None,
        "provider": llm.provider.provider,
        "model": llm.model,
    })
    checkpoint = checkpointer.get_tuple(config)
    saved = checkpoint.checkpoint.get("channel_values", {}) if checkpoint else {}
    if checkpoint and saved.get("fingerprint") != fingerprint:
        raise CheckpointCompatibilityError(
            "Checkpoint is legacy or incompatible with this source/configuration. "
            "Use a fresh --run-id; existing checkpoints were left unchanged."
        )
    doc = build_document_graph(pair, settings, llm)
    graph_fingerprint = _fingerprint({
        "segments": [s.model_dump(mode="json") for s in doc.graph.segments],
        "edges": sorted(e.key() for e in doc.graph.edges),
    })
    if checkpoint and saved.get("graph_fingerprint") != graph_fingerprint:
        raise CheckpointCompatibilityError(
            "Rebuilt graph differs from the checkpoint. Use a fresh --run-id."
        )
    ordered = sorted(doc.graph.segments, key=lambda s: s.order)

    if not ordered:
        return DocumentTranslation(
            doc_id=pair.doc_id,
            direction=pair.direction,
            strategy=strategy_name(settings),
            segments=[],
            records=[],
            output_text="",
            reference_text=pair.reference_text,
            source_text=pair.source_text,
            graph=doc.graph,
            graph_stats=doc.stats,
            stats={
                "n_flagged_segments": 0,
                "n_terminology_candidates": 0,
                "n_terminology_matches": 0,
                "n_unmatched_terminology_candidates": 0,
                "glossary_hit_rate": 0.0,
            },
            glossary_snapshot=glossary.snapshot() if glossary is not None else None,
        )

    segments_by_id = {s.seg_id: s for s in ordered}
    translator = Translator(settings, llm)
    if settings.translation.condition == "graft_baseline":
        builder = _build_memory_app(doc.graph, settings, llm, glossary)
    else:
        builder = _build_graph_app(doc.graph, segments_by_id, settings, translator)
    app = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {"thread_id": f"{pair.direction}:{pair.doc_id}"},
        # Baseline uses one parallel terminology superstep, then two stages per
        # segment: translation and memory extraction.
        "recursion_limit": 2 * len(ordered) + 12,
    }
    # Invoking with a fresh input always (re)starts the thread from scratch,
    # discarding whatever the checkpointer already has for it. Resuming means
    # invoking with `None`, which tells LangGraph to continue the existing
    # thread's state instead — the entire reason a checkpointer was worth
    # adding (D25).
    initial = {"records": {}, "memories": {}, "memory_contexts": {}, "terminology": {},
               "fingerprint": fingerprint, "graph_fingerprint": graph_fingerprint}
    final_state = app.invoke(None if checkpoint else initial, config=config)

    records_by_id = {
        seg_id: TranslationRecord.model_validate(rec)
        for seg_id, rec in final_state["records"].items()
    }
    records = [records_by_id[s.seg_id] for s in ordered]
    output = _reassemble(pair.source_text, ordered, records)

    terminology_records = [
        TerminologyRecord.model_validate(final_state["terminology"][s.seg_id])
        for s in ordered
        if s.seg_id in final_state.get("terminology", {})
    ]
    n_candidates = sum(len(record.candidates) for record in terminology_records)
    n_matches = sum(len(record.matches) for record in terminology_records)
    n_unmatched = sum(len(record.unmatched_candidates) for record in terminology_records)
    lookup_total = n_matches + n_unmatched
    stats = {
        "n_flagged_segments": sum(1 for r in records if r.flags),
        "n_terminology_candidates": n_candidates,
        "n_terminology_matches": n_matches,
        "n_unmatched_terminology_candidates": n_unmatched,
        "glossary_hit_rate": round(n_matches / lookup_total, 4) if lookup_total else 0.0,
    }

    return DocumentTranslation(
        doc_id=pair.doc_id,
        direction=pair.direction,
        strategy=strategy_name(settings),
        segments=ordered,
        records=records,
        output_text=output,
        reference_text=pair.reference_text,
        source_text=pair.source_text,
        graph=doc.graph,
        graph_stats=doc.stats,
        stats=stats,
        memories=[MemoryRecord.model_validate(final_state["memories"][s.seg_id])
                  for s in ordered if s.seg_id in final_state.get("memories", {})],
        memory_contexts=final_state.get("memory_contexts", {}),
        terminology=terminology_records,
        glossary_snapshot=glossary.snapshot() if glossary is not None else None,
    )


def write_document_artifacts(run: RunDir, result: DocumentTranslation) -> None:
    """Persist one document's artifacts so any stage can be re-run alone."""
    base = run.subdir("documents") / result.doc_id
    base.mkdir(parents=True, exist_ok=True)
    run.write_json(f"documents/{result.doc_id}/segments.json", result.segments)
    run.write_json(f"documents/{result.doc_id}/records.json", result.records)
    if result.strategy == "graft_baseline":
        run.write_json(f"documents/{result.doc_id}/memories.json", result.memories)
        run.write_json(f"documents/{result.doc_id}/memory_contexts.json", result.memory_contexts)
    if result.terminology:
        run.write_json(f"documents/{result.doc_id}/terminology.json", result.terminology)
    if result.glossary_snapshot is not None:
        snapshot_path = Path(run.path) / "glossary_snapshot.json"
        if snapshot_path.exists():
            if run.read_json("glossary_snapshot.json") != result.glossary_snapshot:
                raise RuntimeError("run glossary snapshot differs from this document")
        else:
            run.write_json("glossary_snapshot.json", result.glossary_snapshot)
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

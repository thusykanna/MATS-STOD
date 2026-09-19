"""DAG-context condition run, scored and tabulated.

Kept as its own pipeline (rather than folded into `dag_translate.py`) so a
future condition (a different `dag_context_depth`, an ablation on
`graph.transitive_reduction`) can be added here and compared without
touching the single-condition `translate` command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from ..config import Settings
from ..evaluation.metrics import score_corpus, score_document
from ..evaluation.report import _table, models_used
from ..io.parallel import DocPair
from ..io.runs import RunDir
from ..llm.client import CachedLLM
from ..llm.ledger import CallLedger
from .dag_translate import (
    STRATEGY_NAME,
    checkpoint_path,
    translate_document,
    write_document_artifacts,
)


@dataclass
class ConditionResult:
    name: str
    chrf: float
    bleu: float
    n_docs: int
    calls: int
    cache_hits: int
    tokens_in: int
    tokens_out: int
    latency_s: float
    flagged_segments: int
    notes: list[str] = field(default_factory=list)


def run_condition(
    pairs: list[DocPair],
    settings: Settings,
    llm: CachedLLM,
    run: RunDir,
) -> ConditionResult:
    """Run the DAG-context condition and score it."""
    # A fresh ledger slice so this condition's counts are attributed correctly
    # even though it shares the run's cache with anything else in the run.
    before = len(llm.ledger.calls)

    sub = run.subdir(f"conditions/{STRATEGY_NAME}")
    cond_run = RunDir(sub, settings, dry_run=run.dry_run)

    hyps: list[str] = []
    refs: list[str] = []
    doc_scores = []
    flagged = 0

    with SqliteSaver.from_conn_string(str(checkpoint_path(cond_run))) as checkpointer:
        for pair in pairs:
            result = translate_document(pair, settings, llm, checkpointer)
            write_document_artifacts(cond_run, result)
            hyps.append(result.output_text)
            refs.append(result.reference_text)
            doc_scores.append(
                score_document(
                    pair.doc_id, result.output_text, pair.reference_text, settings.evaluation
                )
            )
            flagged += int(result.stats.get("n_flagged_segments", 0))

    corpus = score_corpus(hyps, refs, settings.evaluation)
    slice_ledger = CallLedger(calls=llm.ledger.calls[before:])

    return ConditionResult(
        name=STRATEGY_NAME,
        chrf=corpus.chrf,
        bleu=corpus.bleu,
        n_docs=len(pairs),
        calls=slice_ledger.n_calls,
        cache_hits=slice_ledger.n_cached,
        tokens_in=slice_ledger.tokens_in_total,
        tokens_out=slice_ledger.tokens_out_total,
        latency_s=round(slice_ledger.latency_s_total, 2),
        flagged_segments=flagged,
        notes=[],
    )


def render_comparison(
    results: list[ConditionResult], settings: Settings, model: str | None = None
) -> str:
    rows = [
        [
            r.name,
            r.chrf,
            r.bleu,
            r.calls,
            r.cache_hits,
            r.tokens_in,
            r.tokens_out,
            r.latency_s,
            r.flagged_segments,
        ]
        for r in results
    ]
    body = _table(
        [
            "condition",
            "chrF++",
            "BLEU",
            "calls",
            "cached",
            "tokens in",
            "tokens out",
            "latency s",
            "flagged",
        ],
        rows,
    )
    lines = [
        f"# Condition results ({settings.langs.direction})\n",
        f"Model: {model or settings.llm.model}. "
        f"Prompt: {settings.translation.prompt_version}. "
        f"Documents: {results[0].n_docs if results else 0}.\n",
        body,
    ]
    notes = [n for r in results for n in r.notes]
    if notes:
        lines.append("\n## Notes\n")
        lines.extend(f"- {n}" for n in dict.fromkeys(notes))
        lines.append("")
    return "\n".join(lines)


def run_comparison(
    pairs: list[DocPair],
    settings: Settings,
    llm: CachedLLM,
    run: RunDir,
) -> dict[str, Any]:
    results = [run_condition(pairs, settings, llm, run)]
    # The model that was actually called, not the one named in config; with
    # `--provider fake` those differ and the table must not claim otherwise.
    model = models_used(llm.ledger)
    payload = {
        "direction": settings.langs.direction,
        "model": model,
        "model_configured": settings.llm.model,
        "n_docs": len(pairs),
        "conditions": [r.__dict__ for r in results],
    }
    run.write_json("comparison.json", payload)
    run.write_text("comparison.md", render_comparison(results, settings, model))
    return payload

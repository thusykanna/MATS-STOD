"""Strategy comparison.

Runs several conditions over the same documents with the same model and the
same prompt, and puts the numbers in one table. Anything that differs between
conditions other than the context strategy is a confound, so the loop builds
each condition from the same `Settings` object with only the strategy changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..evaluation.metrics import score_corpus, score_document
from ..evaluation.report import _table
from ..io.parallel import DocPair
from ..io.runs import RunDir
from ..llm.client import CachedLLM
from ..llm.cost import CostLedger
from ..translation.context import build_strategy
from .baseline import translate_document, write_document_artifacts


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
    cost_usd_cold: float
    cost_usd_billed: float
    latency_s: float
    flagged_segments: int
    notes: list[str] = field(default_factory=list)


def run_condition(
    strategy_name: str,
    pairs: list[DocPair],
    settings: Settings,
    llm: CachedLLM,
    run: RunDir,
) -> ConditionResult:
    """Run one condition and score it."""
    strat_settings = settings.model_copy(deep=True)
    strat_settings.translation.strategy = strategy_name
    strategy = build_strategy(strategy_name, strat_settings)

    # A fresh ledger per condition so cost is attributed correctly even though
    # the conditions share one cache.
    before = len(llm.ledger.calls)

    hyps: list[str] = []
    refs: list[str] = []
    doc_scores = []
    notes: list[str] = []
    flagged = 0

    for pair in pairs:
        result = translate_document(pair, strat_settings, llm, strategy)
        sub = run.subdir(f"conditions/{strategy.name}")
        cond_run = RunDir(sub, strat_settings, dry_run=run.dry_run)
        write_document_artifacts(cond_run, result)
        hyps.append(result.output_text)
        refs.append(result.reference_text)
        doc_scores.append(
            score_document(
                pair.doc_id, result.output_text, pair.reference_text, settings.evaluation
            )
        )
        notes.extend(result.notes)
        flagged += int(result.stats.get("n_flagged_segments", 0))

    corpus = score_corpus(hyps, refs, settings.evaluation)
    slice_ledger = CostLedger(
        prices_usd_per_mtok=settings.llm.prices_usd_per_mtok,
        calls=llm.ledger.calls[before:],
    )

    return ConditionResult(
        name=strategy.name,
        chrf=corpus.chrf,
        bleu=corpus.bleu,
        n_docs=len(pairs),
        calls=slice_ledger.n_calls,
        cache_hits=slice_ledger.n_cached,
        tokens_in=slice_ledger.tokens_in_total,
        tokens_out=slice_ledger.tokens_out_total,
        cost_usd_cold=round(slice_ledger.cost_usd_cold, 6),
        cost_usd_billed=round(slice_ledger.cost_usd_billed, 6),
        latency_s=round(slice_ledger.latency_s_total, 2),
        flagged_segments=flagged,
        notes=notes,
    )


def render_comparison(results: list[ConditionResult], settings: Settings) -> str:
    rows = [
        [
            r.name,
            r.chrf,
            r.bleu,
            r.calls,
            r.cache_hits,
            r.tokens_in,
            r.tokens_out,
            r.cost_usd_cold,
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
            "USD cold",
            "latency s",
            "flagged",
        ],
        rows,
    )
    lines = [
        f"# Strategy comparison ({settings.langs.direction})\n",
        f"Model: {settings.llm.model}. Prompt: {settings.translation.prompt_version}. "
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
    strategies: list[str],
    pairs: list[DocPair],
    settings: Settings,
    llm: CachedLLM,
    run: RunDir,
) -> dict[str, Any]:
    results = [run_condition(name, pairs, settings, llm, run) for name in strategies]
    payload = {
        "direction": settings.langs.direction,
        "model": settings.llm.model,
        "n_docs": len(pairs),
        "conditions": [r.__dict__ for r in results],
    }
    run.write_json("comparison.json", payload)
    run.write_text("comparison.md", render_comparison(results, settings))
    return payload

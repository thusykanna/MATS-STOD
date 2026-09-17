"""Run reports.

A run produces `results.json` for machines and `report.md` for the write-up.
Both are deterministic: identical input produces identical bytes, so a diff
between two runs shows only what actually changed. Timestamps live in
`run_meta.json` and the event log, never here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..config import Settings
from ..io.runs import RunDir
from ..llm.cost import CostLedger
from .metrics import CorpusScore, DocScore


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def models_used(ledger: CostLedger | None) -> str:
    """The model or models this run actually called.

    Read from the ledger rather than from config, because the two can differ:
    `--provider fake` leaves the configured model name untouched, and a report
    that named a model it never called would misattribute every number in it.
    """
    if ledger is None or not ledger.calls:
        return "none (no LLM calls in this run)"
    names = sorted({c.model for c in ledger.calls})
    return ", ".join(names)


def build_results(
    settings: Settings,
    doc_scores: list[DocScore],
    corpus: CorpusScore | None,
    ledger: CostLedger | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_name": settings.run_name,
        "direction": settings.langs.direction,
        "strategy": settings.translation.strategy,
        "segmenter": settings.segmentation.segmenter,
        "edge_inferrer": settings.graph.edge_inferrer,
        "model": models_used(ledger),
        "model_configured": settings.llm.model,
        "corpus": asdict(corpus) if corpus else None,
        "documents": [asdict(d) for d in doc_scores],
        "cost": ledger.summary() if ledger else None,
        **(extra or {}),
    }


def render_report(results: dict[str, Any], settings: Settings) -> str:
    """Render report.md from a results dict."""
    lines: list[str] = []
    lines.append(f"# MATS-STOD run: {results.get('run_name', 'unnamed')}\n")

    lines.append("## Condition\n")
    lines.append(
        _table(
            ["setting", "value"],
            [
                ["direction", results.get("direction")],
                ["strategy", results.get("strategy")],
                ["segmenter", results.get("segmenter")],
                ["edge inferrer", results.get("edge_inferrer")],
                ["model actually called", results.get("model")],
                ["model in config", results.get("model_configured")],
                ["primary metric", settings.evaluation.primary_metric],
                ["BLEU tokeniser", settings.evaluation.bleu_tokenize],
            ],
        )
    )

    corpus = results.get("corpus")
    if corpus:
        lines.append("\n## Corpus scores\n")
        lines.append(
            _table(
                ["metric", "score", "documents"],
                [
                    ["chrF++", corpus["chrf"], corpus["n_docs"]],
                    ["BLEU", corpus["bleu"], corpus["n_docs"]],
                ],
            )
        )

    docs = results.get("documents") or []
    if docs:
        lines.append("\n## Per-document scores\n")
        lines.append(
            _table(
                ["doc_id", "chrF++", "BLEU", "hyp chars", "ref chars"],
                [
                    [d["doc_id"], d["chrf"], d["bleu"], d["n_chars_hyp"], d["n_chars_ref"]]
                    for d in docs
                ],
            )
        )

    seg_stats = results.get("segmentation_stats")
    if seg_stats:
        lines.append("\n## Segmentation\n")
        lines.append(
            _table(
                ["statistic", "value"],
                [[k, v] for k, v in seg_stats.items() if not isinstance(v, (dict, list))],
            )
        )

    seg_scores = results.get("segmentation_scores")
    if seg_scores:
        lines.append("\n## Segmentation against gold\n")
        lines.append(
            _table(
                ["doc_id", "P", "R", "F1", "Pk", "WindowDiff"],
                [
                    [
                        s["doc_id"],
                        s["precision"],
                        s["recall"],
                        s["f1"],
                        s["pk"],
                        s["window_diff"],
                    ]
                    for s in seg_scores.get("documents", [])
                ],
            )
        )
        agg = seg_scores.get("corpus")
        if agg:
            lines.append("\nCorpus: " + ", ".join(f"{k}={v}" for k, v in agg.items()) + "\n")

    flags = results.get("flagged_segments")
    if flags:
        lines.append("\n## Flagged segments (digit mismatch, not corrected)\n")
        lines.append(
            _table(
                ["doc_id", "seg_id", "flag"],
                [[f["doc_id"], f["seg_id"], f["flag"]] for f in flags],
            )
        )

    cost = results.get("cost")
    if cost:
        lines.append("\n## Cost\n")
        lines.append(
            _table(
                ["measure", "value"],
                [
                    ["LLM calls", cost["calls"]],
                    ["cache hits", cost["cache_hits"]],
                    ["tokens in (billed)", cost["tokens_in_billed"]],
                    ["tokens out (billed)", cost["tokens_out_billed"]],
                    ["USD spent this run", cost["cost_usd_billed"]],
                    ["USD from a cold cache", cost["cost_usd_cold"]],
                ],
            )
        )
        by_purpose = cost.get("by_purpose") or {}
        if by_purpose:
            lines.append("\nBy purpose:\n")
            lines.append(
                _table(
                    ["purpose", "calls", "cached", "tokens in", "tokens out", "USD cold"],
                    [
                        [
                            k,
                            v["calls"],
                            v["cached"],
                            v["tokens_in"],
                            v["tokens_out"],
                            round(v["cost_usd_cold"], 6),
                        ]
                        for k, v in sorted(by_purpose.items())
                    ],
                )
            )

    notes = results.get("notes") or []
    if notes:
        lines.append("\n## Notes\n")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("\n## Exact configuration used\n")
    lines.append("```yaml\n" + settings.to_yaml() + "```\n")
    return "\n".join(lines)


def write_report(
    run: RunDir,
    settings: Settings,
    doc_scores: list[DocScore],
    corpus: CorpusScore | None,
    ledger: CostLedger | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write results.json and report.md into the run directory."""
    results = build_results(settings, doc_scores, corpus, ledger, extra)
    run.write_json("results.json", results)
    run.write_text("report.md", render_report(results, settings))
    return results

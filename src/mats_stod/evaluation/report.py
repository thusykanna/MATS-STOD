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
from ..llm.ledger import CallLedger
from .metrics import CorpusScore, DocScore


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def models_used(ledger: CallLedger | None) -> str:
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
    ledger: CallLedger | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_name": settings.run_name,
        "direction": settings.langs.direction,
        "model": models_used(ledger),
        "model_configured": settings.llm.model,
        "corpus": asdict(corpus) if corpus else None,
        "documents": [asdict(d) for d in doc_scores],
        "calls": ledger.summary() if ledger else None,
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
        rows = [
            ["chrF++", corpus["chrf"], corpus["n_docs"]],
            ["BLEU", corpus["bleu"], corpus["n_docs"]],
        ]
        if "comet" in (corpus.get("extra") or {}):
            rows.append(["COMET", corpus["extra"]["comet"], corpus["n_docs"]])
        lines.append(_table(["metric", "score", "documents"], rows))

    docs = results.get("documents") or []
    if docs:
        lines.append("\n## Per-document scores\n")
        has_comet = any("comet" in (d.get("extra") or {}) for d in docs)
        headers = ["doc_id", "chrF++", "BLEU", "hyp chars", "ref chars"]
        rows = [
            [d["doc_id"], d["chrf"], d["bleu"], d["n_chars_hyp"], d["n_chars_ref"]]
            + ([d["extra"].get("comet", "")] if has_comet else [])
            for d in docs
        ]
        lines.append(_table(headers + (["COMET"] if has_comet else []), rows))

    input_summary = results.get("input_summary")
    if input_summary:
        if corpus is None:
            lines.append("\nCorpus evaluation: unavailable — no reference files\n")
        lines.append("\n## Translation inputs\n")
        lines.append(
            _table(
                ["measure", "value"],
                [
                    ["input directory", input_summary["input_directory"]],
                    ["discovered folders", input_summary["discovered_folders"]],
                    ["valid folders", input_summary["valid_folders"]],
                    ["selected folders", input_summary["selected_folders"]],
                    ["translated documents", input_summary["translated_documents"]],
                    ["evaluated documents", input_summary["evaluated_documents"]],
                    ["source-only documents", input_summary["source_only_documents"]],
                ],
            )
        )
        skipped = input_summary.get("skipped_folders") or []
        if skipped:
            lines.append("\nSkipped folders:\n")
            lines.append(
                _table(
                    ["folder", "reason"],
                    [[item["doc_id"], item["reason"]] for item in skipped],
                )
            )

    term_eval = results.get("terminology_evaluation")
    if term_eval:
        lines.append("\n## Terminology evaluation\n")
        identification = term_eval["identification"]
        resolution = term_eval["resolution"]
        recovery = term_eval["recovery"]
        realization = term_eval["target_realization"]
        lines.append(_table(
            ["measure", "value"],
            [
                ["method", results.get("terminology_method")],
                ["gold documents", term_eval["documents"]],
                ["identification precision", identification["precision"]],
                ["identification recall", identification["recall"]],
                ["identification F1", identification["f1"]],
                ["resolution precision", resolution["precision"]],
                ["resolution recall", resolution["recall"]],
                ["false glossary mappings", resolution["false_mappings"]],
                ["recovery recall", recovery["recall"]],
                ["target realization accuracy", realization["accuracy"]],
            ],
        ))

    term_ops = results.get("terminology_operational")
    if term_ops:
        lines.append("\n## Terminology operations\n")
        rows = [
            ["method", results.get("terminology_method")],
            ["candidates", term_ops["candidates"]],
            ["approved matches", term_ops["matches"]],
            ["unmatched candidates", term_ops["unmatched"]],
            ["latency s", term_ops["latency_s"]],
        ]
        rows.extend(
            [f"resolved: {method}", count]
            for method, count in term_ops["resolution_methods"].items()
        )
        rows.extend(
            [f"source form: {kind}", count]
            for kind, count in term_ops.get("source_term_kinds", {}).items()
        )
        lines.append(_table(["measure", "value"], rows))

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

    calls = results.get("calls")
    if calls:
        lines.append("\n## LLM calls\n")
        lines.append(
            _table(
                ["measure", "value"],
                [
                    ["logical requests", calls.get("logical_requests", calls["calls"])],
                    ["completed responses", calls["calls"]],
                    ["provider calls", calls.get("provider_calls", "not recorded")],
                    ["cache hits", calls["cache_hits"]],
                    ["tokens in (billed)", calls["tokens_in_billed"]],
                    ["tokens out (billed)", calls["tokens_out_billed"]],
                ],
            )
        )
        lines.append(calls.get("accounting_scope", "") + "\n")
        by_purpose = calls.get("by_purpose") or {}
        if by_purpose:
            lines.append("\nBy purpose:\n")
            lines.append(
                _table(
                    ["purpose", "calls", "cached", "tokens in", "tokens out"],
                    [
                        [k, v["calls"], v["cached"], v["tokens_in"], v["tokens_out"]]
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
    ledger: CallLedger | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write results.json and report.md into the run directory."""
    results = build_results(settings, doc_scores, corpus, ledger, extra)
    run.write_json("results.json", results)
    run.write_text("report.md", render_report(results, settings))
    return results

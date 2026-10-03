"""Command line interface.

Every command takes `--config`, `--max-docs` and `--dry-run`, because the
token budget is a free trial and a command that cannot be limited or costed
before it runs is a command that will eventually spend the budget by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .config import Settings, load_settings
from .evaluation.comet_metric import add_comet_if_enabled
from .evaluation.metrics import score_corpus, score_document
from .evaluation.report import write_report
from .evaluation.terminology import aggregate_terminology_scores
from .io.env import credential_status, load_env_file
from .io.parallel import (
    TranslationDiscovery,
    discover_translation_inputs,
    load_parallel,
    load_text_dir,
)
from .io.runs import RunDir
from .io.split import create_split, select
from .llm.factory import build_llm

app = typer.Typer(
    add_completion=False,
    help="MATS-STOD: Sinhala-Tamil official document translation research pipeline.",
)

# Credentials come from the environment; a gitignored .env at the repository
# root is loaded once here so they need not be exported in every terminal.
load_env_file()


def _settings(
    config: str | None,
    experiment: list[str] | None,
    source: str | None,
    target: str | None,
    model: str | None = None,
    provider: str | None = None,
) -> Settings:
    overrides: dict[str, Any] = {}
    if source or target:
        overrides["langs"] = {}
        if source:
            overrides["langs"]["source"] = source
        if target:
            overrides["langs"]["target"] = target
    if model:
        overrides.setdefault("llm", {})["model"] = model
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    return load_settings(config, overlays=list(experiment or []), overrides=overrides)


def _load_docs(
    settings: Settings,
    data_dir: str | None,
    portion: str,
    max_docs: int | None,
):
    root = data_dir or settings.paths.parallel
    if not Path(root).exists():
        typer.echo(
            f"No data at {root}. Falling back to the committed samples at "
            f"{settings.paths.samples}.",
            err=True,
        )
        root = settings.paths.samples
    doc_ids = None
    split_path = Path(settings.paths.split_file)
    if portion != "all" and split_path.exists():
        available = [p.name for p in sorted(Path(root).iterdir()) if p.is_dir()]
        doc_ids = select(available, split_path, portion)
    pairs = load_parallel(
        root,
        settings.langs.source,
        settings.langs.target,
        doc_ids=doc_ids,
        max_docs=max_docs,
    )
    if not pairs:
        raise typer.BadParameter(
            f"no document pairs for direction {settings.langs.direction} under {root}"
        )
    return pairs


_TRANSLATION_DIRECTIONS = {
    "1": ("si", "ta"),
    "2": ("ta", "si"),
}


def _resolve_translation_direction(
    source: str | None, target: str | None
) -> tuple[str, str]:
    """Resolve explicit flags or ask for one of the two supported directions."""
    if (source is None) != (target is None):
        raise typer.BadParameter("provide both --source and --target, or omit both")
    if source is not None and target is not None:
        if (source, target) not in {("si", "ta"), ("ta", "si")}:
            raise typer.BadParameter(
                "translate supports --source si --target ta or --source ta --target si"
            )
        return source, target

    typer.echo("Select translation direction:\n")
    typer.echo("1. Sinhala → Tamil")
    typer.echo("2. Tamil → Sinhala\n")
    while True:
        choice = typer.prompt("Enter 1 or 2", show_default=False).strip()
        selected = _TRANSLATION_DIRECTIONS.get(choice)
        if selected is not None:
            return selected[0], selected[1]
        typer.echo("Invalid choice. Enter 1 or 2.", err=True)


def _translation_discovery(
    settings: Settings,
    data_dir: str | None,
    portion: str,
    max_docs: int | None,
) -> TranslationDiscovery:
    """Discover default translation inputs without falling back to samples."""
    if portion not in {"all", "dev", "test"}:
        raise typer.BadParameter("portion must be all, dev or test", param_hint="--portion")
    root = Path(data_dir or settings.paths.parallel)
    if not root.exists():
        raise typer.BadParameter(f"parallel data directory not found: {root}")
    if not root.is_dir():
        raise typer.BadParameter(f"parallel data path is not a directory: {root}")

    doc_ids = None
    split_path = Path(settings.paths.split_file)
    if portion != "all":
        if not split_path.exists():
            raise typer.BadParameter(
                f"split file not found: {split_path}. Create it with `mats-stod make-split`."
            )
        available = [path.name for path in sorted(root.iterdir()) if path.is_dir()]
        doc_ids = select(available, split_path, portion)

    discovery = discover_translation_inputs(
        root,
        settings.langs.source,
        settings.langs.target,
        doc_ids=doc_ids,
        max_docs=max_docs,
    )
    if not discovery.pairs:
        reasons = "; ".join(
            f"{item.doc_id}: {item.reason}" for item in discovery.skipped
        )
        detail = f" Skipped folders: {reasons}" if reasons else ""
        raise typer.BadParameter(
            f"no usable {settings.langs.source} source documents under {root}.{detail}"
        )
    return discovery


def _print_translation_discovery(settings: Settings, discovery: TranslationDiscovery) -> None:
    """Print the exact workload before any model client or run is created."""
    typer.echo(
        f"Translation direction: {settings.langs.source_name} → "
        f"{settings.langs.target_name}"
    )
    typer.echo(f"Input directory: {discovery.root}")
    typer.echo(f"Discovered folders: {discovery.discovered_count}")
    typer.echo(f"Valid document folders: {discovery.valid_count}")
    typer.echo(f"Selected for translation: {len(discovery.pairs)}")
    typer.echo(f"Skipped folders: {len(discovery.skipped)}")
    for item in discovery.skipped:
        typer.echo(f"  - {item.doc_id}: {item.reason}", err=True)


# --------------------------------------------------------------------------


@app.command()
def eval(
    hyp: str = typer.Option(..., "--hyp", help="Directory of hypothesis .txt files."),
    ref: str = typer.Option(..., "--ref", help="Directory of reference .txt files."),
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    max_docs: int | None = typer.Option(None, "--max-docs"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Score a directory of translations against references."""
    settings = _settings(config, experiment, source, target)
    hyps = load_text_dir(hyp)
    refs = load_text_dir(ref)
    shared = sorted(set(hyps) & set(refs))
    if not shared:
        raise typer.BadParameter("no file stems are shared between --hyp and --ref")
    if max_docs is not None:
        shared = shared[:max_docs]

    missing = sorted((set(hyps) | set(refs)) - set(shared))
    doc_scores = [score_document(d, hyps[d], refs[d], settings.evaluation) for d in shared]
    corpus = score_corpus([hyps[d] for d in shared], [refs[d] for d in shared], settings.evaluation)

    run = RunDir.create(settings, prefix="eval", run_id=run_id, dry_run=dry_run)
    notes = [f"Scored {len(shared)} documents from {hyp} against {ref}."]
    if missing:
        notes.append(f"Ignored {len(missing)} unmatched files: {missing[:10]}")
    write_report(run, settings, doc_scores, corpus, None, extra={"notes": notes})
    typer.echo(f"chrF++ {corpus.chrf}  BLEU {corpus.bleu}  documents {corpus.n_docs}")
    typer.echo(f"Report: {run.path / 'report.md'}")


@app.command()
def translate(
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    data: str | None = typer.Option(None, "--data", help="Parallel data directory."),
    portion: str = typer.Option("all", "--portion", help="dev, test or all."),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider", help="gemini, openai_compat, fake."),
    max_docs: int | None = typer.Option(None, "--max-docs", min=1),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Run the complete translation pipeline over folders in data/parallel."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    from .pipelines.dag_translate import (
        checkpoint_path,
        translate_document,
        write_document_artifacts,
    )

    source, target = _resolve_translation_direction(source, target)
    settings = _settings(config, experiment, source, target, model=model, provider=provider)
    discovery = _translation_discovery(settings, data, portion, max_docs)
    pairs = discovery.pairs
    _print_translation_discovery(settings, discovery)
    llm = build_llm(settings, dry_run=dry_run)
    run = RunDir.create(settings, prefix="translate-dag", run_id=run_id, dry_run=dry_run)

    hyps, refs, srcs, doc_scores = [], [], [], []
    terminology_scores = []
    terminology_operational = {
        "candidates": 0, "matches": 0, "unmatched": 0, "latency_s": 0.0,
        "resolution_methods": {},
        "source_term_kinds": {},
    }
    with SqliteSaver.from_conn_string(str(checkpoint_path(run))) as checkpointer:
        for pair in pairs:
            try:
                result = translate_document(pair, settings, llm, checkpointer)
            except Exception as exc:
                run.log("document_failed", doc_id=pair.doc_id,
                        condition=settings.translation.condition,
                        error_type=type(exc).__name__, error=str(exc))
                raise
            write_document_artifacts(run, result)
            if result.terminology_evaluation is not None:
                terminology_scores.append(result.terminology_evaluation)
            terminology_operational["candidates"] += result.stats.get(
                "n_terminology_candidates", 0
            )
            terminology_operational["matches"] += result.stats.get("n_terminology_matches", 0)
            terminology_operational["unmatched"] += result.stats.get(
                "n_unmatched_terminology_candidates", 0
            )
            terminology_operational["latency_s"] += result.stats.get(
                "terminology_latency_s", 0.0
            )
            for method_name, count in result.stats.get(
                "terminology_resolution_methods", {}
            ).items():
                methods = terminology_operational["resolution_methods"]
                methods[method_name] = methods.get(method_name, 0) + count
            for kind, count in result.stats.get("terminology_source_term_kinds", {}).items():
                kinds = terminology_operational["source_term_kinds"]
                kinds[kind] = kinds.get(kind, 0) + count
            typer.echo(f"Translated: {pair.doc_id}")
            typer.echo(f"Output: {run.path / 'documents' / pair.doc_id / 'translation.txt'}")
            if result.reference_text is not None:
                score = score_document(
                    pair.doc_id, result.output_text, result.reference_text, settings.evaluation
                )
                hyps.append(result.output_text)
                refs.append(result.reference_text)
                srcs.append(pair.source_text)
                doc_scores.append(score)
                typer.echo(f"Evaluation: chrF++ {score.chrf}  BLEU {score.bleu}")
            else:
                typer.echo(
                    f"Evaluation: skipped — no {settings.langs.target_name} reference file"
                )
            run.log(
                "translated", doc_id=pair.doc_id, segments=len(result.segments),
                reference_available=result.reference_text is not None,
                evaluated=result.reference_text is not None,
            )

    corpus = score_corpus(hyps, refs, settings.evaluation) if doc_scores else None
    if corpus is not None:
        add_comet_if_enabled(settings.evaluation, doc_scores, corpus, srcs, hyps, refs)
    flagged = [
        {"doc_id": p.doc_id, "seg_id": r["seg_id"], "flag": flag}
        for p in pairs
        for r in run.read_json(f"documents/{p.doc_id}/records.json")
        for flag in r["flags"]
    ]
    terminology_active = (
        settings.translation.condition == "graft_baseline" and settings.terminology.enabled
    )
    write_report(
        run,
        settings,
        doc_scores,
        corpus,
        llm.ledger,
        extra={
            "flagged_segments": flagged,
            "terminology_method": settings.terminology.method if terminology_active else None,
            "terminology_evaluation": (
                aggregate_terminology_scores(terminology_scores) if terminology_active else None
            ),
            "terminology_operational": ({
                **terminology_operational,
                "latency_s": round(terminology_operational["latency_s"], 6),
                "resolution_methods": dict(sorted(
                    terminology_operational["resolution_methods"].items()
                )),
                "source_term_kinds": dict(sorted(
                    terminology_operational["source_term_kinds"].items()
                )),
            } if terminology_active else None),
            "input_summary": {
                "input_directory": str(discovery.root),
                "discovered_folders": discovery.discovered_count,
                "valid_folders": discovery.valid_count,
                "selected_folders": len(pairs),
                "translated_documents": len(pairs),
                "evaluated_documents": len(doc_scores),
                "source_only_documents": len(pairs) - len(doc_scores),
                "skipped_folders": [
                    {"doc_id": item.doc_id, "reason": item.reason}
                    for item in discovery.skipped
                ],
            },
        },
    )
    typer.echo(f"Translated documents: {len(pairs)}")
    typer.echo(f"Evaluated documents: {len(doc_scores)}")
    if corpus is None:
        typer.echo("Corpus evaluation: unavailable — no reference files")
    else:
        comet = (corpus.extra or {}).get("comet")
        typer.echo(
            f"Corpus evaluation: chrF++ {corpus.chrf}  BLEU {corpus.bleu}"
            f"  documents {corpus.n_docs}"
        )
        if comet is not None:
            typer.echo(f"COMET score: {comet}")
    typer.echo(f"Report: {run.path / 'report.md'}")


@app.command()
def compare(
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    data: str | None = typer.Option(None, "--data"),
    portion: str = typer.Option("dev", "--portion"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider"),
    max_docs: int | None = typer.Option(None, "--max-docs"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Score and tabulate the configured translation condition."""
    from .pipelines.compare import run_comparison

    settings = _settings(config, experiment, source, target, model=model, provider=provider)
    pairs = _load_docs(settings, data, portion, max_docs)
    llm = build_llm(settings, dry_run=dry_run)
    run = RunDir.create(settings, prefix="compare", run_id=run_id, dry_run=dry_run)

    payload = run_comparison(pairs, settings, llm, run)
    for cond in payload["conditions"]:
        typer.echo(
            f"{cond['name']:<20} chrF++ {cond['chrf']:<8} BLEU {cond['bleu']:<8} "
            f"calls {cond['calls']:<5}"
        )
    typer.echo(f"Table: {run.path / 'comparison.md'}")


@app.command()
def segment(
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    data: str | None = typer.Option(None, "--data"),
    portion: str = typer.Option("dev", "--portion"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider"),
    max_docs: int | None = typer.Option(None, "--max-docs"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Segment documents with GRAFT and score against gold when it exists."""
    from .pipelines.segment_pipeline import run_segmentation

    settings = _settings(config, experiment, source, target, model=model, provider=provider)
    pairs = _load_docs(settings, data, portion, max_docs)
    llm = build_llm(settings, dry_run=dry_run)
    run = RunDir.create(settings, prefix="segment-graft", run_id=run_id, dry_run=dry_run)

    extra = run_segmentation(pairs, settings, llm, run)
    write_report(run, settings, [], None, llm.ledger, extra=extra)
    stats = extra["segmentation_stats"]
    typer.echo(
        f"{stats['n_documents']} documents, {stats['n_segments_total']} segments, "
        f"{stats['llm_calls_total']} LLM calls"
    )
    typer.echo(f"Report: {run.path / 'report.md'}")


@app.command("build-graph")
def build_graph(
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    data: str | None = typer.Option(None, "--data"),
    portion: str = typer.Option("dev", "--portion"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider"),
    max_docs: int | None = typer.Option(None, "--max-docs"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Build a discourse graph per document with GRAFT and score against gold when it exists."""
    from .pipelines.graph_pipeline import run_graph_build

    settings = _settings(config, experiment, source, target, model=model, provider=provider)
    pairs = _load_docs(settings, data, portion, max_docs)
    llm = build_llm(settings, dry_run=dry_run)
    run = RunDir.create(settings, prefix="graph-graft", run_id=run_id, dry_run=dry_run)

    extra = run_graph_build(pairs, settings, llm, run)
    write_report(run, settings, [], None, llm.ledger, extra=extra)

    stats = extra["graph_stats"]
    typer.echo(
        f"{stats['n_documents']} documents, {stats['n_edges_total']} edges, "
        f"{stats['llm_calls_total']} LLM calls"
    )
    # The number that decides whether a DAG condition can beat a sliding
    # window at all, printed where it cannot be missed.
    typer.echo(
        f"adjacent-only edges {stats['adjacent_only_share']:.1%}  |  "
        f"segments with a non-adjacent parent {stats['share_with_nonadjacent_parent']:.1%}"
    )
    typer.echo(f"Report: {run.path / 'report.md'}")


@app.command("annotate-template")
def annotate_template(
    doc: str = typer.Argument(..., help="Path to a pair directory or a source .txt file."),
    out: str = typer.Option("data/gold/templates", "--out"),
    config: str | None = typer.Option(None, "--config"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
) -> None:
    """Write a human-editable gold annotation template for one document."""
    from .evaluation.annotate import write_template
    from .io.parallel import load_pair_dir
    from .io.text import read_text
    from .parsing.plaintext import PlainTextParser

    settings = _settings(config, None, source, target)
    path = Path(doc)
    if path.is_dir():
        pairs = load_pair_dir(path, [(settings.langs.source, settings.langs.target)])
        if not pairs:
            raise typer.BadParameter(f"no {settings.langs.direction} pair in {path}")
        document = PlainTextParser().parse(
            pairs[0].source_text, pairs[0].doc_id, pairs[0].source_lang
        )
    else:
        document = PlainTextParser().parse(read_text(path), path.stem, settings.langs.source)

    written = write_template(document, settings, out)
    typer.echo(f"Template: {written}")
    typer.echo("Mark segment starts with B in column 2, add edges under EDGES, then run:")
    typer.echo(f"  mats-stod annotate-import {written}")


@app.command("annotate-import")
def annotate_import(
    path: str = typer.Argument(..., help="A filled-in .annot.tsv file."),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Convert a filled-in annotation template into gold JSON files."""
    from .evaluation.annotate import import_annotation

    settings = _settings(config, None, None, None)
    seg_path, edge_path = import_annotation(
        path, settings.paths.gold_segmentation, settings.paths.gold_edges
    )
    typer.echo(f"Gold segmentation: {seg_path}")
    typer.echo(f"Gold edges: {edge_path}")


@app.command("make-split")
def make_split(
    data: str | None = typer.Option(None, "--data"),
    dev_fraction: float = typer.Option(0.5, "--dev-fraction"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Create the fixed dev/test split. Never reshuffles an existing one."""
    settings = _settings(config, None, None, None)
    root = Path(data or settings.paths.parallel)
    if not root.exists():
        root = Path(settings.paths.samples)
    doc_ids = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    split = create_split(
        doc_ids, settings.paths.split_file, dev_fraction, settings.seed, overwrite
    )
    typer.echo(json.dumps(split, ensure_ascii=False, indent=2))


@app.command("compare-terminology")
def compare_terminology(
    runs: list[str] = typer.Argument(..., help="Two or more completed run directories."),
    out: str = typer.Option("terminology-comparison", "--out"),
) -> None:
    """Validate and compare completed terminology runs without rerunning them."""
    from .evaluation.terminology_compare import (
        compare_terminology_runs,
        render_terminology_comparison,
    )

    payload = compare_terminology_runs(runs)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(
        render_terminology_comparison(payload), encoding="utf-8"
    )
    typer.echo(f"Comparison: {output / 'comparison.md'}")


@app.command("show-config")
def show_config(
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
) -> None:
    """Print the fully resolved configuration."""
    typer.echo(_settings(config, experiment, source, target).to_yaml())


@app.command("check-llm")
def check_llm(
    config: str | None = typer.Option(None, "--config"),
    send: bool = typer.Option(
        False, "--send", help="Actually send one tiny test call (costs a few tokens)."
    ),
) -> None:
    """Diagnose LLM credentials without spending tokens.

    Run this before any real translation: it reports what is configured, what
    the environment provides, and, with --send, whether one minimal call
    actually succeeds.
    """
    settings = _settings(config, None, None, None)
    typer.echo(f"provider : {settings.llm.provider}")
    typer.echo(f"backend  : {settings.llm.backend}")
    typer.echo(f"model    : {settings.llm.model}")
    typer.echo("")
    typer.echo("Environment:")
    for name, state in credential_status().items():
        typer.echo(f"  {name:<32} {state}")
    typer.echo("")

    try:
        from .llm.factory import build_provider

        provider = build_provider(settings)
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        typer.echo(f"Client could not be created: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"Client created: {provider.provider} / {provider.model}")

    if not send:
        typer.echo("")
        typer.echo("No call was made. Re-run with --send to test a real call.")
        return

    from .llm.base import Message

    try:
        response = provider.complete(
            [Message("user", "Reply with the single word: ok")],
            max_output_tokens=8,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Call failed: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Call succeeded. Reply: {response.text[:60]!r} "
        f"(tokens in {response.tokens_in}, out {response.tokens_out})"
    )


def main() -> None:  # pragma: no cover
    """Entry point that reports expected conditions without a traceback.

    A dry run finding uncached work, or a missing credential, is a normal
    outcome of a guard doing its job. Printing a stack trace for it makes a
    working safeguard look like a crash.
    """
    from .graph.graft_edges import PairwiseCallBudgetError
    from .io.runs import RunConfigurationError
    from .llm.base import LLMError
    from .llm.client import DryRunExhausted
    from .pipelines.dag_translate import CheckpointCompatibilityError
    from .pipelines.graph_pipeline import BaselineConfigurationError

    # SystemExit rather than typer.Exit: this is outside the Typer callback,
    # where typer.Exit is not translated into an exit code.
    try:
        app()
    except DryRunExhausted as exc:
        typer.echo(f"\nDry run stopped: {exc}", err=True)
        typer.echo(
            "Nothing was sent and nothing was charged. Re-run without --dry-run "
            "to make these calls.",
            err=True,
        )
        raise SystemExit(2) from None
    except (PairwiseCallBudgetError, CheckpointCompatibilityError, RunConfigurationError,
            BaselineConfigurationError) as exc:
        typer.echo(f"\nRun stopped: {exc}", err=True)
        raise SystemExit(2) from None
    except LLMError as exc:
        typer.echo(f"\nLLM call failed: {exc}", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()

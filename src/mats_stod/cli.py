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
from .evaluation.metrics import score_corpus, score_document
from .evaluation.report import write_report
from .io.env import credential_status, load_env_file
from .io.parallel import load_parallel, load_text_dir
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
    strategy: str | None = None,
    segmenter: str | None = None,
    edges: str | None = None,
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
    if strategy:
        overrides.setdefault("translation", {})["strategy"] = strategy
    if segmenter:
        overrides.setdefault("segmentation", {})["segmenter"] = segmenter
    if edges:
        overrides.setdefault("graph", {})["edge_inferrer"] = edges
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
    strategy: str = typer.Option("B1", "--strategy", help="B0, B1 or B2."),
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    data: str | None = typer.Option(None, "--data", help="Parallel data directory."),
    portion: str = typer.Option("dev", "--portion", help="dev, test or all."),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider", help="gemini, openai_compat, fake."),
    max_docs: int | None = typer.Option(None, "--max-docs"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Translate documents under one baseline context strategy."""
    from .pipelines.baseline import translate_document, write_document_artifacts

    settings = _settings(
        config, experiment, source, target, strategy=strategy, model=model, provider=provider
    )
    pairs = _load_docs(settings, data, portion, max_docs)
    llm = build_llm(settings, dry_run=dry_run)
    run = RunDir.create(settings, prefix=f"translate-{strategy}", run_id=run_id, dry_run=dry_run)

    hyps, refs, doc_scores, notes = [], [], [], []
    for pair in pairs:
        result = translate_document(pair, settings, llm)
        write_document_artifacts(run, result)
        hyps.append(result.output_text)
        refs.append(result.reference_text)
        doc_scores.append(
            score_document(
                pair.doc_id, result.output_text, pair.reference_text, settings.evaluation
            )
        )
        notes.extend(result.notes)
        run.log("translated", doc_id=pair.doc_id, segments=len(result.segments))

    corpus = score_corpus(hyps, refs, settings.evaluation)
    flagged = [
        {"doc_id": p.doc_id, "seg_id": r["seg_id"], "flag": flag}
        for p in pairs
        for r in run.read_json(f"documents/{p.doc_id}/records.json")
        for flag in r["flags"]
    ]
    write_report(
        run,
        settings,
        doc_scores,
        corpus,
        llm.ledger,
        extra={"notes": notes, "flagged_segments": flagged},
    )
    typer.echo(f"chrF++ {corpus.chrf}  BLEU {corpus.bleu}  documents {corpus.n_docs}")
    typer.echo(f"Report: {run.path / 'report.md'}")


@app.command()
def compare(
    strategies: str = typer.Option("B0,B1,B2", "--strategies"),
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
    """Run several strategies on the same documents and tabulate them."""
    from .pipelines.compare import run_comparison

    settings = _settings(config, experiment, source, target, model=model, provider=provider)
    names = [s.strip() for s in strategies.split(",") if s.strip()]
    pairs = _load_docs(settings, data, portion, max_docs)
    llm = build_llm(settings, dry_run=dry_run)
    run = RunDir.create(settings, prefix="compare", run_id=run_id, dry_run=dry_run)

    payload = run_comparison(names, pairs, settings, llm, run)
    for cond in payload["conditions"]:
        typer.echo(
            f"{cond['name']:<20} chrF++ {cond['chrf']:<8} BLEU {cond['bleu']:<8} "
            f"calls {cond['calls']:<5} USD(cold) {cond['cost_usd_cold']}"
        )
    typer.echo(f"Table: {run.path / 'comparison.md'}")


@app.command()
def segment(
    segmenter: str = typer.Option("structural", "--segmenter", help="structural or graft."),
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
    """Segment documents and score against gold when it exists."""
    from .pipelines.segment_pipeline import run_segmentation

    settings = _settings(
        config, experiment, source, target, segmenter=segmenter, model=model, provider=provider
    )
    pairs = _load_docs(settings, data, portion, max_docs)
    llm = build_llm(settings, dry_run=dry_run) if segmenter == "graft" else None
    run = RunDir.create(settings, prefix=f"segment-{segmenter}", run_id=run_id, dry_run=dry_run)

    extra = run_segmentation(pairs, settings, llm, run)
    write_report(run, settings, [], None, llm.ledger if llm else None, extra=extra)
    stats = extra["segmentation_stats"]
    typer.echo(
        f"{stats['n_documents']} documents, {stats['n_segments_total']} segments, "
        f"{stats['llm_calls_total']} LLM calls"
    )
    typer.echo(f"Report: {run.path / 'report.md'}")


@app.command("build-graph")
def build_graph(
    edges: str = typer.Option(
        "graft", "--edges", help="graft, predecessor, tfidf or none."
    ),
    config: str | None = typer.Option(None, "--config"),
    experiment: list[str] | None = typer.Option(None, "--experiment"),
    data: str | None = typer.Option(None, "--data"),
    portion: str = typer.Option("dev", "--portion"),
    source: str | None = typer.Option(None, "--source"),
    target: str | None = typer.Option(None, "--target"),
    segmenter: str | None = typer.Option(None, "--segmenter"),
    model: str | None = typer.Option(None, "--model"),
    provider: str | None = typer.Option(None, "--provider"),
    max_docs: int | None = typer.Option(None, "--max-docs"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Build a discourse graph per document and score against gold when it exists."""
    from .pipelines.graph_pipeline import run_graph_build

    settings = _settings(
        config,
        experiment,
        source,
        target,
        segmenter=segmenter,
        edges=edges,
        model=model,
        provider=provider,
    )
    pairs = _load_docs(settings, data, portion, max_docs)
    # Only the LLM-backed inferrers and the graft segmenter need a client; the
    # deterministic ones must stay runnable with no credentials at all.
    needs_llm = (
        settings.graph.edge_inferrer == "graft" or settings.segmentation.segmenter == "graft"
    )
    llm = build_llm(settings, dry_run=dry_run) if needs_llm else None
    run = RunDir.create(settings, prefix=f"graph-{edges}", run_id=run_id, dry_run=dry_run)

    extra = run_graph_build(pairs, settings, llm, run)
    write_report(run, settings, [], None, llm.ledger if llm else None, extra=extra)

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
    from .llm.base import LLMError
    from .llm.client import DryRunExhausted

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
    except LLMError as exc:
        typer.echo(f"\nLLM call failed: {exc}", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()

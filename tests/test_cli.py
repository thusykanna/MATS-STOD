"""End-to-end command tests, all offline through the fake provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mats_stod.cli import app

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config pointing every output at a temporary directory."""
    cfg = tmp_path / "config.yaml"
    base = yaml_text(tmp_path)
    cfg.write_text(base, encoding="utf-8")
    return cfg


def yaml_text(tmp_path: Path) -> str:
    import yaml

    from mats_stod.config import load_settings

    s = load_settings()
    s.paths.runs = str(tmp_path / "runs")
    s.paths.gold_segmentation = str(tmp_path / "gold_seg")
    s.paths.gold_edges = str(tmp_path / "gold_edges")
    s.paths.gold_terminology = str(tmp_path / "gold_terminology")
    s.paths.split_file = str(tmp_path / "splits.json")
    s.llm.cache_path = str(tmp_path / "cache.sqlite")
    s.llm.provider = "fake"
    s.evaluation.comet_enabled = False
    return yaml.safe_dump(s.model_dump(mode="json"), sort_keys=True, allow_unicode=True)


def run(*args: str):
    result = runner.invoke(app, list(args))
    if result.exit_code != 0:
        raise AssertionError(f"command failed: {args}\n{result.output}\n{result.exception}")
    return result


def test_show_config_resolves_overlays(workspace):
    out = run("show-config", "--config", str(workspace), "--source", "ta", "--target", "si").output
    assert "source: ta" in out and "target: si" in out


def test_make_split_then_reuse(workspace, tmp_path):
    run("make-split", "--config", str(workspace), "--data", "data/samples")
    split = json.loads((tmp_path / "splits.json").read_text())
    expected = {p.name for p in Path("data/samples").iterdir() if p.is_dir()}
    assert set(split["dev"]) | set(split["test"]) == expected
    assert not set(split["dev"]) & set(split["test"])


def test_translate_runs_the_dag_condition(workspace, tmp_path):
    import shutil

    corpus = tmp_path / "explicit-corpus"
    shutil.copytree("data/samples/circular_01", corpus / "circular_01")
    run(
        "translate", "--config", str(workspace),
        "--provider", "fake", "--data", str(corpus), "--portion", "all",
        "--source", "si", "--target", "ta",
        "--max-docs", "1", "--run-id", "t-dag",
    )
    results = json.loads((tmp_path / "runs" / "t-dag" / "results.json").read_text())
    assert results["corpus"]["n_docs"] == 1
    assert results["calls"]["calls"] > 0
    base = tmp_path / "runs" / "t-dag" / "documents" / "circular_01"
    records = json.loads((base / "records.json").read_text())
    memories = json.loads((base / "memories.json").read_text())
    contexts = json.loads((base / "memory_contexts.json").read_text())
    terminology = json.loads((base / "terminology.json").read_text())
    stats = json.loads((base / "stats.json").read_text())["stats"]
    glossary = json.loads((tmp_path / "runs" / "t-dag" / "glossary_snapshot.json").read_text())
    assert len(records) == len(memories) == len(contexts) == len(terminology)
    assert all(r["context_strategy"] == "graft_baseline" for r in records)
    assert "memory_extraction" in results["calls"]["by_purpose"]
    assert "terminology_extraction" in results["calls"]["by_purpose"]
    assert glossary["version"] == "dummy-government-v2-gazette" and glossary["sha256"]
    assert stats["n_terminology_matches"] > 0
    assert any(record["metadata"]["terminology_entry_ids"] for record in records)


def test_translate_menu_option_one_selects_sinhala_to_tamil(workspace, tmp_path):
    result = runner.invoke(
        app,
        [
            "translate", "--config", str(workspace), "--provider", "fake",
            "--data", "data/samples", "--max-docs", "1", "--run-id", "menu-si-ta",
        ],
        input="1\n",
    )

    assert result.exit_code == 0, result.output
    assert "1. Sinhala → Tamil" in result.output
    assert "Translation direction: Sinhala → Tamil" in result.output
    payload = json.loads((tmp_path / "runs" / "menu-si-ta" / "results.json").read_text())
    assert payload["direction"] == "si-ta"


def test_default_translation_discovers_source_only_and_paired_folders(
    workspace, tmp_path, monkeypatch,
):
    import yaml

    parallel = tmp_path / "parallel"
    source_only = parallel / "a_source_only"
    paired = parallel / "b_paired"
    wrong_direction = parallel / "c_wrong_direction"
    for directory in (source_only, paired, wrong_direction):
        directory.mkdir(parents=True)
    (source_only / "notice.ta").write_text("அரச சுற்றறிக்கை.", encoding="utf-8")
    (paired / "notice.ta").write_text("நிதி அமைச்சு.", encoding="utf-8")
    (paired / "notice.si").write_text("මුදල් අමාත්‍යාංශය.", encoding="utf-8")
    (wrong_direction / "notice.si").write_text("චක්‍රලේඛය.", encoding="utf-8")

    config = yaml.safe_load(workspace.read_text(encoding="utf-8"))
    config["paths"]["parallel"] = str(parallel)
    config["evaluation"]["comet_enabled"] = True
    workspace.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    comet_inputs = []

    def fake_score(self, sources, hypotheses, references):
        comet_inputs.append((sources, hypotheses, references))
        return [0.75] * len(sources)

    monkeypatch.setattr("mats_stod.evaluation.comet_metric.CometMetric.score", fake_score)

    result = runner.invoke(
        app,
        [
            "translate", "--config", str(workspace), "--provider", "fake",
            "--run-id", "mixed-default",
        ],
        input="invalid\n2\n",
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert "Invalid choice. Enter 1 or 2." in result.output
    assert "Translation direction: Tamil → Sinhala" in result.output
    assert "Valid document folders: 2" in result.output
    assert "Selected for translation: 2" in result.output
    assert "c_wrong_direction" in result.output
    assert "Evaluation: skipped — no Sinhala reference file" in result.output
    assert "Translated documents: 2" in result.output
    assert "Evaluated documents: 1" in result.output
    assert "COMET score: 0.75" in result.output

    run_dir = tmp_path / "runs" / "mixed-default"
    assert (run_dir / "documents" / "a_source_only" / "translation.txt").exists()
    assert (run_dir / "documents" / "b_paired" / "translation.txt").exists()
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["corpus"]["n_docs"] == 1
    assert payload["input_summary"]["source_only_documents"] == 1
    assert payload["input_summary"]["evaluated_documents"] == 1
    assert payload["input_summary"]["skipped_folders"][0]["doc_id"] == "c_wrong_direction"
    assert payload["corpus"]["extra"]["comet"] == 0.75
    assert len(comet_inputs) == 1
    sources, hypotheses, references = comet_inputs[0]
    assert sources == ["நிதி அமைச்சு."]
    assert references == ["මුදල් අමාත්‍යාංශය."]
    assert len(hypotheses) == 1


def test_source_only_run_writes_translation_without_corpus_scores(workspace, tmp_path, monkeypatch):
    import yaml

    config = yaml.safe_load(workspace.read_text())
    config["evaluation"]["comet_enabled"] = True
    workspace.write_text(yaml.safe_dump(config))

    def unexpected_comet(*args, **kwargs):
        pytest.fail("COMET must not be invoked without references")

    monkeypatch.setattr("mats_stod.cli.add_comet_if_enabled", unexpected_comet)
    source_dir = tmp_path / "source-only" / "notice"
    source_dir.mkdir(parents=True)
    (source_dir / "notice.ta").write_text("அரச சுற்றறிக்கை.", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "translate", "--config", str(workspace), "--provider", "fake",
            "--data", str(source_dir.parent), "--source", "ta", "--target", "si",
            "--run-id", "source-only",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert "Corpus evaluation: unavailable — no reference files" in result.output
    run_dir = tmp_path / "runs" / "source-only"
    assert (run_dir / "documents" / "notice" / "translation.txt").exists()
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["corpus"] is None
    assert payload["documents"] == []
    assert payload["input_summary"]["source_only_documents"] == 1
    assert payload["input_summary"]["evaluated_documents"] == 0
    assert "no reference files" in (run_dir / "report.md").read_text()


@pytest.mark.parametrize("portion", ["dev", "test", "invalid"])
def test_translate_rejects_missing_split_or_invalid_portion(workspace, tmp_path, portion):
    result = runner.invoke(app, [
        "translate", "--config", str(workspace), "--data", "data/samples",
        "--source", "si", "--target", "ta", "--portion", portion,
    ])
    assert result.exit_code != 0
    message = "portion must be" if portion == "invalid" else "make-split"
    assert message in result.output
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("portion, expected", [("dev", "a"), ("test", "b")])
def test_translate_uses_requested_split(workspace, tmp_path, portion, expected):
    corpus = tmp_path / "corpus"
    for name in ("a", "b"):
        folder = corpus / name
        folder.mkdir(parents=True)
        (folder / "doc.si").write_text("මුදල් අමාත්‍යාංශය.", encoding="utf-8")
    (tmp_path / "splits.json").write_text(json.dumps({"dev": ["a"], "test": ["b"]}))
    run(
        "translate", "--config", str(workspace), "--data", str(corpus),
        "--source", "si", "--target", "ta", "--portion", portion, "--run-id", portion,
    )
    documents = tmp_path / "runs" / portion / "documents"
    assert {p.name for p in documents.iterdir()} == {expected}


def test_translate_missing_default_parallel_directory_does_not_fall_back(workspace, tmp_path):
    import yaml

    config = yaml.safe_load(workspace.read_text(encoding="utf-8"))
    config["paths"]["parallel"] = str(tmp_path / "missing-parallel")
    workspace.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "translate", "--config", str(workspace), "--provider", "fake",
            "--source", "si", "--target", "ta",
        ],
    )

    assert result.exit_code != 0
    assert "parallel data directory not found" in result.output
    assert "Falling back" not in result.output
    assert not (tmp_path / "runs").exists()


def test_translate_prompt_cancellation_creates_no_run(workspace, tmp_path):
    result = runner.invoke(
        app,
        ["translate", "--config", str(workspace), "--provider", "fake"],
        input="",
    )

    assert result.exit_code != 0
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "flags",
    [
        ("--source", "si"),
        ("--source", "en", "--target", "ta"),
    ],
)
def test_translate_rejects_incomplete_or_unsupported_direction(workspace, tmp_path, flags):
    result = runner.invoke(
        app,
        ["translate", "--config", str(workspace), "--provider", "fake", *flags],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_translate_rejects_nonpositive_max_docs(workspace, tmp_path, limit):
    result = runner.invoke(
        app,
        [
            "translate", "--config", str(workspace), "--provider", "fake",
            "--source", "si", "--target", "ta", "--max-docs", limit,
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("direction", [("si", "ta"), ("ta", "si")])
def test_translate_runs_in_both_directions(workspace, tmp_path, direction):
    src, tgt = direction
    run(
        "translate", "--config", str(workspace),
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--source", src, "--target", tgt, "--max-docs", "1",
        "--run-id", f"dir-{src}{tgt}",
    )
    results = json.loads((tmp_path / "runs" / f"dir-{src}{tgt}" / "results.json").read_text())
    assert results["direction"] == f"{src}-{tgt}"


def test_compare_produces_one_table_with_comet(workspace, tmp_path, monkeypatch):
    import yaml

    config = yaml.safe_load(workspace.read_text())
    config["evaluation"]["comet_enabled"] = True
    workspace.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "mats_stod.evaluation.comet_metric.CometMetric.score",
        lambda self, sources, hypotheses, references: [0.61] * len(sources),
    )
    run(
        "compare", "--config", str(workspace),
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--max-docs", "1", "--run-id", "cmp",
    )
    table = (tmp_path / "runs" / "cmp" / "comparison.md").read_text()
    assert "graft_baseline" in table
    payload = json.loads((tmp_path / "runs" / "cmp" / "comparison.json").read_text())
    assert len(payload["conditions"]) == 1
    assert payload["conditions"][0]["comet"] == 0.61
    assert "COMET" in table


def test_segment_runs_with_graft(workspace, tmp_path):
    run(
        "segment", "--config", str(workspace),
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--run-id", "seg-graft",
    )
    results = json.loads((tmp_path / "runs" / "seg-graft" / "results.json").read_text())
    stats = results["segmentation_stats"]
    expected_count = sum(p.is_dir() for p in Path("data/samples").iterdir())
    assert stats["n_documents"] == expected_count and stats["n_segments_total"] > 0
    assert stats["llm_calls_total"] > 0


def test_eval_scores_a_directory(workspace, tmp_path):
    hyp, ref = tmp_path / "hyp", tmp_path / "ref"
    hyp.mkdir(); ref.mkdir()
    (hyp / "a.txt").write_text("ஒரே உரை இங்கே உள்ளது", encoding="utf-8")
    (ref / "a.txt").write_text("ஒரே உரை இங்கே உள்ளது", encoding="utf-8")
    out = run("eval", "--config", str(workspace), "--hyp", str(hyp), "--ref", str(ref)).output
    assert "chrF++ 100.0" in out


def test_rerunning_is_free_because_of_the_cache(workspace, tmp_path):
    args = (
        "translate", "--config", str(workspace),
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--source", "si", "--target", "ta",
        "--max-docs", "1",
    )
    run(*args, "--run-id", "cold")
    run(*args, "--run-id", "warm")
    cold = json.loads((tmp_path / "runs" / "cold" / "results.json").read_text())["calls"]
    warm = json.loads((tmp_path / "runs" / "warm" / "results.json").read_text())["calls"]
    assert cold["cache_hits"] == 0
    assert warm["cache_hits"] == warm["calls"]
    assert warm["tokens_in_billed"] == 0


def test_dry_run_refuses_to_spend_tokens(workspace, tmp_path):
    result = runner.invoke(
        app,
        ["translate", "--config", str(workspace), "--provider", "fake",
         "--data", "data/samples", "--portion", "all", "--max-docs", "1",
         "--source", "si", "--target", "ta",
         "--dry-run", "--run-id", "dry"],
    )
    assert result.exit_code != 0
    assert "dry run" in str(result.exception).lower()


def test_annotate_template_and_import(workspace, tmp_path):
    out = run(
        "annotate-template", "data/samples/circular_01",
        "--config", str(workspace), "--out", str(tmp_path / "templates"),
    ).output
    template = tmp_path / "templates" / "circular_01.si.annot.tsv"
    assert template.exists() and "annotate-import" in out
    run("annotate-import", str(template), "--config", str(workspace))
    assert (tmp_path / "gold_seg" / "circular_01.si.json").exists()


def test_segment_scores_against_gold_when_it_exists(workspace, tmp_path):
    run(
        "annotate-template", "data/samples/circular_01",
        "--config", str(workspace), "--out", str(tmp_path / "templates"),
    )
    run("annotate-import", str(tmp_path / "templates" / "circular_01.si.annot.tsv"),
        "--config", str(workspace))
    run(
        "segment", "--config", str(workspace),
        "--data", "data/samples", "--portion", "all", "--run-id", "seg-gold",
    )
    results = json.loads((tmp_path / "runs" / "seg-gold" / "results.json").read_text())
    assert "segmentation_scores" in results
    assert results["segmentation_scores"]["documents"][0]["doc_id"] == "circular_01"


def test_gold_for_one_language_is_not_scored_against_the_other(workspace, tmp_path):
    """Sinhala gold must never be applied to a Tamil source document.

    The two sides of a pair are different documents with different
    segmentations; scoring one against the other produced a meaningless F1 of
    zero before gold files were tagged with their source language.
    """
    run(
        "annotate-template", "data/samples/circular_01", "--config", str(workspace),
        "--source", "si", "--target", "ta", "--out", str(tmp_path / "templates"),
    )
    run("annotate-import", str(tmp_path / "templates" / "circular_01.si.annot.tsv"),
        "--config", str(workspace))

    # Segmenting the Tamil side must find no gold and say so.
    run(
        "segment", "--config", str(workspace),
        "--data", "data/samples", "--portion", "all",
        "--source", "ta", "--target", "si", "--run-id", "seg-ta",
    )
    ta = json.loads((tmp_path / "runs" / "seg-ta" / "results.json").read_text())
    assert "segmentation_scores" not in ta
    assert any("No gold segmentation" in n for n in ta["notes"])

    # The Sinhala side, which the gold describes, is scored.
    run(
        "segment", "--config", str(workspace),
        "--data", "data/samples", "--portion", "all",
        "--source", "si", "--target", "ta", "--run-id", "seg-si",
    )
    si = json.loads((tmp_path / "runs" / "seg-si" / "results.json").read_text())
    assert si["segmentation_scores"]["documents"][0]["doc_id"] == "circular_01"


def test_comparison_table_names_the_fake_provider(workspace, tmp_path):
    """The table must say what it actually called."""
    run(
        "compare", "--config", str(workspace),
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--max-docs", "1", "--run-id", "cmp-model",
    )
    table = (tmp_path / "runs" / "cmp-model" / "comparison.md").read_text()
    assert "Model: fake" in table
    assert "gemini" not in table


def _vertex_workspace(workspace, monkeypatch):
    """A config pointing at Vertex with no credentials in the environment."""
    for name in ["GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    import yaml

    cfg = yaml.safe_load(workspace.read_text())
    cfg["llm"]["provider"] = "gemini"
    cfg["llm"]["backend"] = "vertex"
    cfg["llm"]["project"] = None
    workspace.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return workspace


def test_check_llm_reports_missing_credentials(workspace, monkeypatch):
    """The diagnosis must be actionable and must not need a credential.

    Which message appears depends on whether the optional Gemini extra is
    installed, and both are correct; the suite must pass either way, because
    `uv sync --dev` alone does not install that extra.
    """
    cfg = _vertex_workspace(workspace, monkeypatch)
    result = runner.invoke(app, ["check-llm", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "GOOGLE_CLOUD_PROJECT" in result.output
    assert any(
        hint in result.output
        for hint in ("gcloud auth application-default login", "uv sync --extra gemini")
    ), result.output


def test_check_llm_names_the_project_variable_when_the_sdk_is_present(workspace, monkeypatch):
    """With the SDK installed, the missing piece is the project, and it says so."""
    pytest.importorskip("google.genai", reason="needs the optional gemini extra")
    cfg = _vertex_workspace(workspace, monkeypatch)
    result = runner.invoke(app, ["check-llm", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "gcloud auth application-default login" in result.output


def test_check_llm_tells_you_to_install_the_extra_when_it_is_missing(workspace, monkeypatch):
    """Without the SDK, the first actionable step is installing it."""
    import importlib

    try:
        importlib.import_module("google.genai")
    except ImportError:
        pass
    else:
        pytest.skip("the gemini extra is installed, so this path cannot be reached")
    cfg = _vertex_workspace(workspace, monkeypatch)
    result = runner.invoke(app, ["check-llm", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "uv sync --extra gemini" in result.output


def test_check_llm_succeeds_with_the_fake_provider(workspace):
    out = run("check-llm", "--config", str(workspace)).output
    assert "Client created: fake / fake" in out
    assert "No call was made" in out


def test_check_llm_never_prints_a_key(workspace, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-value-here")
    out = run("check-llm", "--config", str(workspace)).output
    assert "secret-key-value-here" not in out
    assert "set (21 chars)" in out


def test_main_reports_edge_budget_without_traceback(monkeypatch, capsys):
    from mats_stod import cli
    from mats_stod.graph.graft_edges import PairwiseCallBudgetError

    def blocked():
        raise PairwiseCallBudgetError('planned 4851, ceiling 4000; set max_pair_distance')

    monkeypatch.setattr(cli, 'app', blocked)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert 'Run stopped' in stderr and 'max_pair_distance' in stderr
    assert 'Traceback' not in stderr

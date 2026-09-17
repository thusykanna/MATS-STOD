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
    s.paths.split_file = str(tmp_path / "splits.json")
    s.llm.cache_path = str(tmp_path / "cache.sqlite")
    s.llm.provider = "fake"
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
    assert set(split["dev"]) | set(split["test"]) == {"circular_01", "notice_02", "memo_03"}


@pytest.mark.parametrize("strategy", ["B0", "B1", "B2"])
def test_translate_each_strategy(workspace, tmp_path, strategy):
    run(
        "translate", "--config", str(workspace), "--strategy", strategy,
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--max-docs", "1", "--run-id", f"t-{strategy}",
    )
    results = json.loads((tmp_path / "runs" / f"t-{strategy}" / "results.json").read_text())
    assert results["corpus"]["n_docs"] == 1
    assert results["cost"]["calls"] > 0


@pytest.mark.parametrize("direction", [("si", "ta"), ("ta", "si")])
def test_translate_runs_in_both_directions(workspace, tmp_path, direction):
    src, tgt = direction
    run(
        "translate", "--config", str(workspace), "--strategy", "B1",
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--source", src, "--target", tgt, "--max-docs", "1",
        "--run-id", f"dir-{src}{tgt}",
    )
    results = json.loads((tmp_path / "runs" / f"dir-{src}{tgt}" / "results.json").read_text())
    assert results["direction"] == f"{src}-{tgt}"


def test_compare_produces_one_table(workspace, tmp_path):
    run(
        "compare", "--config", str(workspace), "--strategies", "B0,B1,B2",
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--max-docs", "1", "--run-id", "cmp",
    )
    table = (tmp_path / "runs" / "cmp" / "comparison.md").read_text()
    for name in ["B0_full_document", "B1_isolated", "B2_sliding_window"]:
        assert name in table
    payload = json.loads((tmp_path / "runs" / "cmp" / "comparison.json").read_text())
    assert len(payload["conditions"]) == 3


@pytest.mark.parametrize("segmenter", ["structural", "graft"])
def test_segment_each_segmenter(workspace, tmp_path, segmenter):
    run(
        "segment", "--config", str(workspace), "--segmenter", segmenter,
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--run-id", f"seg-{segmenter}",
    )
    results = json.loads((tmp_path / "runs" / f"seg-{segmenter}" / "results.json").read_text())
    stats = results["segmentation_stats"]
    assert stats["n_documents"] == 3 and stats["n_segments_total"] > 0
    if segmenter == "graft":
        assert stats["llm_calls_total"] > 0
    else:
        assert stats["llm_calls_total"] == 0


def test_eval_scores_a_directory(workspace, tmp_path):
    hyp, ref = tmp_path / "hyp", tmp_path / "ref"
    hyp.mkdir(); ref.mkdir()
    (hyp / "a.txt").write_text("ஒரே உரை இங்கே உள்ளது", encoding="utf-8")
    (ref / "a.txt").write_text("ஒரே உரை இங்கே உள்ளது", encoding="utf-8")
    out = run("eval", "--config", str(workspace), "--hyp", str(hyp), "--ref", str(ref)).output
    assert "chrF++ 100.0" in out


def test_rerunning_is_free_because_of_the_cache(workspace, tmp_path):
    args = (
        "translate", "--config", str(workspace), "--strategy", "B2",
        "--provider", "fake", "--data", "data/samples", "--portion", "all",
        "--max-docs", "1",
    )
    run(*args, "--run-id", "cold")
    run(*args, "--run-id", "warm")
    cold = json.loads((tmp_path / "runs" / "cold" / "results.json").read_text())["cost"]
    warm = json.loads((tmp_path / "runs" / "warm" / "results.json").read_text())["cost"]
    assert cold["cache_hits"] == 0
    assert warm["cache_hits"] == warm["calls"]
    assert warm["tokens_in_billed"] == 0 and warm["cost_usd_billed"] == 0


def test_dry_run_refuses_to_spend_tokens(workspace, tmp_path):
    result = runner.invoke(
        app,
        ["translate", "--config", str(workspace), "--strategy", "B1", "--provider", "fake",
         "--data", "data/samples", "--portion", "all", "--max-docs", "1",
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
        "segment", "--config", str(workspace), "--segmenter", "structural",
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
        "segment", "--config", str(workspace), "--segmenter", "structural",
        "--data", "data/samples", "--portion", "all",
        "--source", "ta", "--target", "si", "--run-id", "seg-ta",
    )
    ta = json.loads((tmp_path / "runs" / "seg-ta" / "results.json").read_text())
    assert "segmentation_scores" not in ta
    assert any("No gold segmentation" in n for n in ta["notes"])

    # The Sinhala side, which the gold describes, is scored.
    run(
        "segment", "--config", str(workspace), "--segmenter", "structural",
        "--data", "data/samples", "--portion", "all",
        "--source", "si", "--target", "ta", "--run-id", "seg-si",
    )
    si = json.loads((tmp_path / "runs" / "seg-si" / "results.json").read_text())
    assert si["segmentation_scores"]["documents"][0]["doc_id"] == "circular_01"


def test_comparison_table_names_the_fake_provider(workspace, tmp_path):
    """The table must say what it actually called."""
    run(
        "compare", "--config", str(workspace), "--strategies", "B1",
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

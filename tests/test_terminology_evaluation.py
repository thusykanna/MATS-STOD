"""Terminology gold scoring and completed-run comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mats_stod.cli import app
from mats_stod.evaluation.terminology import (
    GoldTerminologyDocument,
    aggregate_terminology_scores,
    load_terminology_gold,
    score_terminology,
)
from mats_stod.evaluation.terminology_compare import (
    TerminologyComparisonError,
    compare_terminology_runs,
)
from mats_stod.schemas import Segment, TranslationRecord
from mats_stod.terminology import GlossaryStore
from mats_stod.terminology.models import TerminologyRecord


def test_gold_scoring_covers_recovery_and_target_realization(tmp_path):
    source = "මුදල් අමාත්‍යාංශයට සහ නව පදය"
    surface = "මුදල් අමාත්‍යාංශයට"
    segment = Segment(
        seg_id="s0", doc_id="d", order=0, text=source, char_start=0, char_end=len(source)
    )
    terminology = TerminologyRecord(
        seg_id="s0",
        extraction_method="llm_lookup_form",
        candidates=[{"surface": surface, "lookup_form": "මුදල් අමාත්‍යාංශය",
                     "reason": "institution"}],
        matches=[{
            "entry_id": "gov-002", "definition": "finance ministry",
            "source_surface": surface, "lookup_form": "මුදල් අමාත්‍යාංශය",
            "source_preferred": "මුදල් අමාත්‍යාංශය", "target_preferred": "நிதி அமைச்சு",
            "char_start": 0, "char_end": len(surface),
            "resolution_methods": ["llm_lookup_form"], "source_term_kind": "preferred",
        }],
        unmatched_candidates=[], glossary_version="v", glossary_hash="h", model="fake",
        prompt_version="terminology_v1",
    )
    translation = TranslationRecord(
        seg_id="s0", source_text=source, target_text="நிதி அமைச்சுக்கு அறிவிக்கவும்",
        context_strategy="graft_baseline", model="fake", prompt_version="translate/v1",
    )
    gold = GoldTerminologyDocument.model_validate({
        "doc_id": "d", "language": "si", "terms": [
            {"surface": surface, "char_start": 0, "char_end": len(surface),
             "entry_id": "gov-002", "accepted_target_forms": ["நிதி அமைச்சுக்கு"]},
            {"surface": "නව පදය", "char_start": source.index("නව පදය"),
             "char_end": len(source), "entry_id": None},
        ],
    })
    score = score_terminology(
        gold, source, [segment], [terminology], [translation],
        GlossaryStore.load("data/glossaries/dummy_government.si-ta.json"), "si", "ta",
    )
    assert score["identification"]["recall"] == 0.5
    assert score["resolution"]["precision"] == 1.0
    assert score["recovery"] == {"correct": 1, "eligible": 1, "recall": 1.0}
    assert score["target_realization"]["accuracy"] == 1.0
    aggregate = aggregate_terminology_scores([score, score])
    assert aggregate["documents"] == 2
    assert aggregate["recovery"]["correct"] == 2


def test_gold_loader_validates_identity_and_source_span(tmp_path):
    root = tmp_path / "gold"
    root.mkdir()
    path = root / "d.si.json"
    path.write_text(json.dumps({
        "doc_id": "d", "language": "si",
        "terms": [{"surface": "wrong", "char_start": 0, "char_end": 5}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="span does not match"):
        load_terminology_gold(root, "d", "si", "source")


def _write_comparison_run(root: Path, name: str, method: str, graph_value: int = 1) -> Path:
    run = root / name
    document = run / "documents" / "d"
    document.mkdir(parents=True)
    config = {
        "langs": {"source": "si", "target": "ta"},
        "llm": {"model": "fake"},
        "segmentation": {"prompt_version": "discourse/v1"},
        "graph": {"prompt_version": "edge/v1"},
        "translation": {"prompt_version": "translate/v1"},
        "terminology": {"prompt_version": "terminology_v1", "method": method},
    }
    (run / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "glossary_snapshot.json").write_text(json.dumps({"sha256": "same"}))
    (document / "graph.json").write_text(json.dumps({"value": graph_value}))
    results = {
        "direction": "si-ta",
        "corpus": {"chrf": 1.0, "bleu": 2.0, "extra": {"comet": 0.5}},
        "terminology_evaluation": {
            "identification": {"f1": 0.4},
            "resolution": {"precision": 0.5, "recall": 0.6, "false_mappings": 1},
            "recovery": {"recall": 0.7}, "target_realization": {"accuracy": 0.8},
        },
        "calls": {"by_purpose": {"terminology_extraction": {
            "calls": 1, "cached": 0, "tokens_in": 10, "tokens_out": 5,
        }}},
    }
    (run / "results.json").write_text(json.dumps(results))
    return run


def test_completed_run_comparison_and_compatibility_checks(tmp_path):
    e0 = _write_comparison_run(tmp_path, "e0", "llm_exact")
    e1 = _write_comparison_run(tmp_path, "e1", "python_scan")
    payload = compare_terminology_runs([e0, e1])
    assert [row["method"] for row in payload["conditions"]] == ["llm_exact", "python_scan"]
    assert payload["conditions"][0]["comet"] == 0.5

    incompatible = _write_comparison_run(tmp_path, "bad", "hybrid", graph_value=2)
    with pytest.raises(TerminologyComparisonError, match="graphs"):
        compare_terminology_runs([e0, incompatible])


def test_compare_terminology_command_writes_json_and_markdown(tmp_path):
    e0 = _write_comparison_run(tmp_path, "e0", "llm_exact")
    e1 = _write_comparison_run(tmp_path, "e1", "python_scan")
    output = tmp_path / "comparison"
    result = CliRunner().invoke(
        app, ["compare-terminology", str(e0), str(e1), "--out", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert (output / "comparison.json").exists()
    assert "llm_exact" in (output / "comparison.md").read_text()

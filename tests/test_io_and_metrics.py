"""Loader, split, metrics and report determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mats_stod.evaluation.metrics import paired_bootstrap, score_corpus, score_document
from mats_stod.evaluation.report import render_report, write_report
from mats_stod.io.parallel import load_pair_dir, load_parallel
from mats_stod.io.runs import RunDir
from mats_stod.io.split import SplitError, create_split, select

# -- loader ---------------------------------------------------------------


def test_one_pair_directory_yields_both_directions():
    pairs = load_pair_dir(Path("data/samples/circular_01"), [("si", "ta"), ("ta", "si")])
    assert len(pairs) == 2
    forward, backward = pairs
    assert forward.direction == "si-ta" and backward.direction == "ta-si"
    assert forward.source_text == backward.reference_text
    assert forward.reference_text == backward.source_text


def test_pair_key_is_unique_per_direction():
    pairs = load_pair_dir(Path("data/samples/circular_01"), [("si", "ta"), ("ta", "si")])
    assert pairs[0].key != pairs[1].key


def test_loading_is_ordered_so_max_docs_is_stable():
    a = [p.doc_id for p in load_parallel("data/samples", "si", "ta", max_docs=2)]
    b = [p.doc_id for p in load_parallel("data/samples", "si", "ta", max_docs=2)]
    assert a == b == sorted(a)


def test_missing_alignment_is_tolerated():
    pair = load_parallel("data/samples", "si", "ta", doc_ids=["circular_01"])[0]
    assert pair.alignment is None


def test_alignment_is_read_when_present(tmp_path):
    d = tmp_path / "doc_a"
    d.mkdir()
    (d / "doc_a.si").write_text("එකයි. දෙකයි.", encoding="utf-8")
    (d / "doc_a.ta").write_text("ஒன்று. இரண்டு.", encoding="utf-8")
    (d / "alignment.jsonl").write_text(
        json.dumps({"si": "එකයි.", "ta": "ஒன்று."}, ensure_ascii=False) + "\n"
        + json.dumps({"si": "දෙකයි.", "ta": "இரண்டு."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pair = load_pair_dir(d, [("si", "ta")])[0]
    assert pair.alignment and len(pair.alignment) == 2
    assert pair.alignment[0].source_text == "එකයි."


def test_source_reference_layout_requires_meta(tmp_path):
    d = tmp_path / "doc_b"
    d.mkdir()
    (d / "source.txt").write_text("a", encoding="utf-8")
    (d / "reference.txt").write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="meta.json"):
        load_pair_dir(d, [("si", "ta")])


def test_source_reference_layout_with_meta(tmp_path):
    d = tmp_path / "doc_c"
    d.mkdir()
    (d / "source.txt").write_text("a", encoding="utf-8")
    (d / "reference.txt").write_text("b", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"source_lang": "si", "target_lang": "ta"}), encoding="utf-8")
    pair = load_pair_dir(d, [("si", "ta")])[0]
    assert pair.source_text == "a" and pair.reference_text == "b"


# -- split ----------------------------------------------------------------


def test_split_is_reproducible(tmp_path):
    ids = [f"doc{i}" for i in range(10)]
    a = create_split(ids, tmp_path / "s1.json", seed=1)
    b = create_split(ids, tmp_path / "s2.json", seed=1)
    assert a["dev"] == b["dev"] and a["test"] == b["test"]
    assert not set(a["dev"]) & set(a["test"])


def test_split_refuses_to_overwrite(tmp_path):
    p = tmp_path / "s.json"
    create_split(["a", "b"], p)
    with pytest.raises(SplitError, match="already exists"):
        create_split(["a", "b"], p)


def test_select_returns_only_the_requested_portion(tmp_path):
    p = tmp_path / "s.json"
    split = create_split([f"d{i}" for i in range(6)], p)
    assert select([f"d{i}" for i in range(6)], p, "dev") == split["dev"]


def test_select_complains_about_missing_documents(tmp_path):
    p = tmp_path / "s.json"
    create_split(["a", "b", "c", "d"], p)
    with pytest.raises(SplitError, match="missing"):
        select(["a"], p, "dev")


# -- metrics --------------------------------------------------------------


def test_identical_text_scores_perfectly(settings):
    text = "ஒரே உரை இங்கே உள்ளது இன்று"
    s = score_document("d", text, text, settings.evaluation)
    assert s.chrf == 100.0 and s.bleu == 100.0


def test_bleu_is_zero_on_text_shorter_than_four_tokens(settings):
    """A documented limitation, not a defect.

    BLEU needs 4-grams, so with whitespace tokenisation any string under four
    tokens scores 0 even when it is identical to the reference. Sinhala and
    Tamil are morphologically rich, so short segments hit this often. It is
    the concrete reason chrF++ is the primary metric and BLEU is secondary.
    """
    s = score_document("d", "ஒரே உரை", "ஒரே உரை", settings.evaluation)
    assert s.chrf == 100.0
    assert s.bleu == 0.0


def test_unrelated_text_scores_poorly(settings):
    s = score_document("d", "முற்றிலும் வேறு", "ஒரே உரை", settings.evaluation)
    assert s.chrf < 30


def test_metrics_are_deterministic(settings):
    a = score_document("d", "abc def", "abc xyz", settings.evaluation)
    b = score_document("d", "abc def", "abc xyz", settings.evaluation)
    assert a == b


def test_corpus_score_requires_matching_lengths(settings):
    with pytest.raises(ValueError):
        score_corpus(["a"], ["a", "b"], settings.evaluation)


def test_paired_bootstrap_is_seeded_and_reproducible(settings):
    base = ["ஒரு", "இரண்டு", "மூன்று", "நான்கு"]
    better = ["ஒரு உரை", "இரண்டு உரை", "மூன்று உரை", "நான்கு உரை"]
    refs = ["ஒரு உரை", "இரண்டு உரை", "மூன்று உரை", "நான்கு உரை"]
    a = paired_bootstrap(base, better, refs, settings.evaluation)
    b = paired_bootstrap(base, better, refs, settings.evaluation)
    assert a == b
    assert a["delta"] > 0 and a["p_value"] < 0.5


# -- report ---------------------------------------------------------------


def test_report_is_byte_identical_for_identical_input(settings, tmp_path):
    scores = [score_document("d1", "a", "a", settings.evaluation)]
    corpus = score_corpus(["a"], ["a"], settings.evaluation)
    r1 = RunDir.create(settings, "t", run_id="r1")
    r2 = RunDir.create(settings, "t", run_id="r2")
    write_report(r1, settings, scores, corpus, None)
    write_report(r2, settings, scores, corpus, None)
    assert (r1.path / "report.md").read_text() == (r2.path / "report.md").read_text()
    assert (r1.path / "results.json").read_text() == (r2.path / "results.json").read_text()


def test_report_embeds_the_exact_config(settings):
    text = render_report({"run_name": "x"}, settings)
    assert "```yaml" in text
    assert f"source: {settings.langs.source}" in text


def test_run_dir_records_config_and_git(settings):
    run = RunDir.create(settings, "t", run_id="meta1")
    assert (run.path / "config_used.yaml").exists()
    meta = json.loads((run.path / "run_meta.json").read_text())
    assert meta["direction"] == settings.langs.direction


def test_report_names_the_model_actually_called_not_the_configured_one(settings):
    """A report must never attribute numbers to a model it did not call.

    `--provider fake` leaves `llm.model` in the config untouched, so reading
    the model from config printed a Gemini model name on runs that never
    reached Gemini.
    """
    from mats_stod.evaluation.report import build_results
    from mats_stod.llm.cost import CostLedger

    settings.llm.model = "gemini-2.0-flash-001"
    ledger = CostLedger()
    ledger.record("translation", "fake", 10, 5, cached=False)

    results = build_results(settings, [], None, ledger)
    assert results["model"] == "fake"
    assert results["model_configured"] == "gemini-2.0-flash-001"


def test_report_says_so_when_no_model_was_called(settings):
    from mats_stod.evaluation.report import build_results

    results = build_results(settings, [], None, None)
    assert "no LLM calls" in results["model"]

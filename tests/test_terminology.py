"""Offline coverage for terminology extraction and glossary lookup."""

from __future__ import annotations

import json
import unicodedata

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from mats_stod.config import load_settings
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import EchoLLM, FakeLLM
from mats_stod.pipelines import dag_translate
from mats_stod.schemas import DiscourseGraph, Edge, Segment
from mats_stod.terminology import GlossaryStore, TerminologyAgent, TerminologyExtractionError
from mats_stod.terminology.glossary import normalize_term
from mats_stod.terminology.models import TerminologyRecord
from mats_stod.translation.context import ContextBlock
from mats_stod.translation.translator import Translator


def segment(text: str, seg_id: str = "s0") -> Segment:
    return Segment(
        seg_id=seg_id,
        doc_id="d",
        order=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )


def graph() -> DiscourseGraph:
    first = segment("මුදල් අමාත්‍යාංශය", "s0")
    second = Segment(
        seg_id="s1",
        doc_id="d",
        order=1,
        text="අනුමත දීමනාව",
        char_start=20,
        char_end=35,
    )
    return DiscourseGraph(
        doc_id="d",
        segments=[first, second],
        edges=[Edge(src="s0", dst="s1", type="continuation")],
    )


def test_dummy_glossary_loads_and_looks_up_both_directions():
    store = GlossaryStore.load("data/glossaries/dummy_government.si-ta.json")
    si_ta = store.lookup("  මුදල්   අමාත්‍යාංශය ", "si", "ta", "government")
    ta_si = store.lookup("நிதி அமைச்சு", "ta", "si", "government")

    assert si_ta is not None and si_ta.target_preferred == "நிதி அமைச்சு"
    assert ta_si is not None and ta_si.target_preferred == "මුදල් අමාත්‍යාංශය"
    assert store.lookup("මුදල් අමාත්‍යාංශය", "si", "ta", "medical") is None
    assert normalize_term("e\u0301") == normalize_term("é")


@pytest.mark.parametrize("source,target,term", [
    ("si", "ta", "මුදල් අමාත්‍යාංශය"),
    ("ta", "si", "நிதி அமைச்சு"),
])
@pytest.mark.parametrize("prefix", ["Straße ", "İ ", "e\u0301 "])
def test_scan_preserves_original_offsets(source, target, term, prefix):
    store = GlossaryStore.load("data/glossaries/dummy_government.si-ta.json")
    text = prefix + term + " / " + term
    matches = store.find_in_text(text, source, target)
    assert len(matches) == 2
    assert [m.char_start for m in matches] == [len(prefix), len(prefix) + len(term) + 3]
    for match in matches:
        assert match.char_end <= len(text)
        assert text[match.char_start:match.char_end] == match.source_surface == term


def test_glossary_normalizes_forms_before_scanning():
    from mats_stod.terminology.models import GlossaryData

    term = "கொடி"
    data = GlossaryData.model_validate({
        "version": "test", "domain": "government", "entries": [{
            "id": "one", "definition": "test", "terms": {
                "si": {"preferred": "පදය"},
                "ta": {"preferred": unicodedata.normalize("NFD", term),
                       "aliases": [unicodedata.normalize("NFD", "கோடி")]},
            },
        }],
    })
    store = GlossaryStore(data, "test-hash")
    assert data.entries[0].terms["ta"].preferred == term
    matches = store.find_in_text(term + " கோடி", "ta", "si")
    assert [(m.source_surface, m.source_term_kind) for m in matches] == [
        (term, "preferred"), ("கோடி", "alias"),
    ]


@pytest.mark.parametrize("overlay,condition,enabled,depth,parents", [
    (None, "graft_baseline", True, 2, 0),
    ("baseline_no_terminology", "graft_baseline", False, 2, 0),
    ("direct_raw_context", "dag_raw_context", False, 1, 0),
    ("raw_context", "dag_raw_context", False, 2, 4),
])
def test_experiment_settings_preserve_conditions(overlay, condition, enabled, depth, parents):
    overlays = [f"configs/experiments/{overlay}.yaml"] if overlay else []
    settings = load_settings(overlays=overlays)
    assert settings.translation.condition == condition
    assert settings.terminology.enabled is enabled
    assert settings.translation.dag_context_depth == depth
    assert settings.graph.max_parents == parents


@pytest.mark.parametrize("experiment,method", [
    ("terminology_e0_llm_exact", "llm_exact"),
    ("terminology_e1_python_scan", "python_scan"),
    ("terminology_e2_llm_lookup_form", "llm_lookup_form"),
    ("terminology_e3_hybrid", "hybrid"),
])
def test_terminology_experiment_overlays(experiment, method):
    settings = load_settings(overlays=[f"configs/experiments/{experiment}.yaml"])
    assert settings.run_name == experiment
    assert settings.translation.condition == "graft_baseline"
    assert settings.terminology.enabled is True
    assert settings.terminology.method == method


def test_default_terminology_method_and_comet():
    settings = load_settings()
    assert settings.terminology.method == "hybrid"
    assert settings.evaluation.comet_enabled is True


@pytest.mark.parametrize(
    "surface",
    [
        "ගැසට් පත්‍රයෙහි",
        "ශ්‍රී ලංකා ප්‍රජාතාන්ත්‍රික සමාජවාදී ජනරජයේ ගැසට් පත්‍රයේ",
        "තනතුරු - ඇබෑර්තු",
        "විභාග",
        "ටෙන්ඩර්",
        "වෙන්දේසි",
        "දෙපාර්තමේන්තු",
        "සංස්ථා",
        "මණ්ඩල",
        "රජයේ මුද්‍රණාලයට",
        "ඉලෙක්ට්‍රොනික ගනුදෙනු පනත",
        "වගන්තිය",
        "ප්‍රකාශනයක්",
        "රීතියක්",
        "නියෝගයක්",
        "නියමයක්",
        "අතුරු ව්‍යවස්ථාවක්",
        "නිවේදනයක්",
        "ගැසට් පත්‍රයේ පළකළ",
        "පනතකින්",
        "නීති ප්‍රඥප්තියකින්",
        "විධිවිධාන සලස්වා",
        "ඉලෙක්ට්‍රොනික ස්වරූපයේ වන ගැසට් පත්‍රයක පළකරනු ලැබුවහොත්",
        "විධිවිධානය සම්පූර්ණ කර ඇත්තාක් සේ සැලකිය යුතුය",
        "රජයේ මුද්‍රණාලයාධිපති",
        "රජයේ මුද්‍රණ දෙපාර්තමේන්තුවේ",
        "ගැසට් පත්‍රය",
        "වෙබ් අඩවියෙන්",
    ],
)
def test_gazette_run_extracted_terms_are_resolved(surface):
    store = GlossaryStore.load("data/glossaries/dummy_government.si-ta.json")

    assert store.lookup(surface, "si", "ta", "government") is not None


def test_alias_scan_keeps_the_longest_overlapping_form():
    store = GlossaryStore.load("data/glossaries/dummy_government.si-ta.json")
    text = "මෙම චක්‍රලේඛය අද සිට ක්‍රියාත්මක වේ."
    matches = store.find_in_text(text, "si", "ta")

    assert len(matches) == 1
    assert matches[0].source_surface == "චක්‍රලේඛය"
    assert matches[0].target_preferred == "சுற்றறிக்கை"
    assert text[matches[0].char_start:matches[0].char_end] == matches[0].source_surface


@pytest.mark.parametrize("duplicate_id", [False, True])
def test_glossary_rejects_ambiguous_keys_and_duplicate_ids(tmp_path, duplicate_id):
    entries = [
        {
            "id": "one",
            "definition": "first",
            "terms": {
                "si": {"preferred": "පදය", "aliases": []},
                "ta": {"preferred": "சொல் ஒன்று", "aliases": []},
            },
        },
        {
            "id": "one" if duplicate_id else "two",
            "definition": "second",
            "terms": {
                "si": {"preferred": "වෙනත්" if duplicate_id else "පදය", "aliases": []},
                "ta": {"preferred": "சொல் இரண்டு", "aliases": []},
            },
        },
    ]
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {"version": "v1", "domain": "government", "notice": "test", "entries": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    error = (ValidationError, ValueError) if duplicate_id else ValueError
    with pytest.raises(error):
        GlossaryStore.load(path)


def test_deterministic_scan_finds_known_term_when_agent_returns_none(settings):
    settings.terminology.enabled = True
    settings.llm.use_cache = False
    store = GlossaryStore.from_settings(settings)
    llm = build_llm(settings, provider=FakeLLM(default='{"terms": []}'))
    record = TerminologyAgent(settings, llm, store).extract(
        segment("මුදල් අමාත්‍යාංශය නිවේදනය කරයි.")
    )

    assert record.candidates == []
    assert [(m.entry_id, m.target_preferred) for m in record.matches] == [
        ("gov-002", "நிதி அமைச்சு"),
        ("gov-028", "அறிவிப்பு"),
    ]
    assert llm.ledger.by_purpose()["terminology_extraction"]["calls"] == 1


def test_agent_candidates_drive_lookup_and_unmatched_terms_are_audit_only(settings):
    settings.terminology.enabled = True
    settings.llm.use_cache = False
    source = "ගමන් වියදම් ප්‍රතිපූරණය සහ විශේෂ පදය"
    response = {
        "terms": [
            {"surface": "ගමන් වියදම් ප්‍රතිපූරණය",
             "lookup_form": "ගමන් වියදම් ප්‍රතිපූරණය", "reason": "administrative term"},
            {"surface": "විශේෂ පදය", "lookup_form": "විශේෂ පදය",
             "reason": "possible specialist term"},
        ]
    }
    llm = build_llm(settings, provider=FakeLLM(default=json.dumps(response, ensure_ascii=False)))
    record = TerminologyAgent(settings, llm, GlossaryStore.from_settings(settings)).extract(
        segment(source)
    )

    assert [match.entry_id for match in record.matches] == ["gov-005"]
    assert [item.surface_form for item in record.unmatched_candidates] == ["විශේෂ පදය"]
    assert all(pair["source"] != "විශේෂ පදය" for pair in record.prompt_payload())


def _candidate_response(surface, lookup_form):
    return json.dumps({"terms": [{
        "surface": surface, "lookup_form": lookup_form, "reason": "government term",
    }]}, ensure_ascii=False)


def test_e0_uses_surface_only_and_e2_recovers_lookup_form(settings):
    source = "මුදල් අමාත්‍යාංශයට දන්වන්න."
    response = _candidate_response("මුදල් අමාත්‍යාංශයට", "මුදල් අමාත්‍යාංශය")
    settings.llm.use_cache = False
    store = GlossaryStore.from_settings(settings)

    settings.terminology.method = "llm_exact"
    e0 = TerminologyAgent(
        settings, build_llm(settings, provider=FakeLLM(default=response)), store
    ).extract(segment(source))
    assert e0.matches == []
    assert [item.surface_form for item in e0.unmatched_candidates] == [
        "මුදල් අමාත්‍යාංශයට"
    ]

    settings.terminology.method = "llm_lookup_form"
    e2 = TerminologyAgent(
        settings, build_llm(settings, provider=FakeLLM(default=response)), store
    ).extract(segment(source))
    assert len(e2.matches) == 1
    assert e2.matches[0].entry_id == "gov-002"
    assert e2.matches[0].source_surface == "මුදල් අමාත්‍යාංශයට"
    assert e2.matches[0].lookup_form == "මුදල් අමාත්‍යාංශය"
    assert e2.matches[0].resolution_methods == ["llm_lookup_form"]


def test_e2_rejects_hallucinated_lookup_form(settings):
    settings.terminology.method = "llm_lookup_form"
    settings.llm.use_cache = False
    source = "විශේෂ පදයට"
    response = _candidate_response(source, "ග්ලොසරියේ නැති පදය")
    record = TerminologyAgent(
        settings,
        build_llm(settings, provider=FakeLLM(default=response)),
        GlossaryStore.from_settings(settings),
    ).extract(segment(source))
    assert record.matches == []
    assert [item.surface_form for item in record.unmatched_candidates] == [source]


def test_e1_has_no_llm_call_and_finds_repeated_terms(settings):
    settings.terminology.method = "python_scan"
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=FakeLLM(default="must not be called"))
    term = "මුදල් අමාත්‍යාංශය"
    record = TerminologyAgent(settings, llm, GlossaryStore.from_settings(settings)).extract(
        segment(f"{term}; {term}")
    )
    matches = [match for match in record.matches if match.entry_id == "gov-002"]
    assert len(matches) == 2
    assert all(match.resolution_methods == ["python_substring"] for match in matches)
    assert record.tokens_in == record.tokens_out == 0
    assert record.prompt == ""
    assert llm.ledger.calls == []


def test_e3_combines_provenance_for_same_span(settings):
    settings.terminology.method = "hybrid"
    settings.llm.use_cache = False
    term = "මුදල් අමාත්‍යාංශය"
    llm = build_llm(settings, provider=FakeLLM(default=_candidate_response(term, term)))
    record = TerminologyAgent(settings, llm, GlossaryStore.from_settings(settings)).extract(
        segment(term)
    )
    assert len(record.matches) == 1
    assert record.matches[0].resolution_methods == ["llm_surface_exact", "python_substring"]


def test_llm_methods_share_cached_candidate_response(settings):
    term = "මුදල් අමාත්‍යාංශය"
    response = _candidate_response(term, term)
    settings.llm.use_cache = True
    llm = build_llm(settings, provider=FakeLLM(default=response))
    store = GlossaryStore.from_settings(settings)
    settings.terminology.method = "llm_exact"
    first = TerminologyAgent(settings, llm, store).extract(segment(term))
    settings.terminology.method = "llm_lookup_form"
    second = TerminologyAgent(settings, llm, store).extract(segment(term))
    assert first.prompt == second.prompt
    assert first.cached is False
    assert second.cached is True
    assert llm.provider.call_count == 1


def test_non_substring_candidates_retry_then_fail(settings):
    settings.terminology.enabled = True
    settings.terminology.retry_on_parse_failure = 1
    settings.llm.use_cache = False
    bad = json.dumps({"terms": [{"surface": "invented", "lookup_form": "invented",
                                  "reason": "not present"}]})
    llm = build_llm(settings, provider=FakeLLM(responses=[bad, bad]))

    with pytest.raises(TerminologyExtractionError, match="not an exact source substring"):
        TerminologyAgent(settings, llm, GlossaryStore.from_settings(settings)).extract(
            segment("source text")
        )
    assert llm.ledger.by_purpose()["terminology_extraction"]["calls"] == 2


def test_translation_prompt_gives_glossary_precedence_over_memory(settings):
    settings.translation.condition = "graft_baseline"
    llm = build_llm(settings, provider=EchoLLM())
    term = TerminologyRecord(
        seg_id="s0",
        extraction_method="hybrid",
        glossary_version="v1",
        glossary_hash="hash",
        model="fake",
        prompt_version="terminology_v1",
        matches=[
            {
                "entry_id": "gov-002",
                "definition": "finance ministry",
                "source_surface": "මුදල් අමාත්‍යාංශය",
                "lookup_form": "මුදල් අමාත්‍යාංශය",
                "source_preferred": "මුදල් අමාත්‍යාංශය",
                "target_preferred": "நிதி அமைச்சு",
                "char_start": 0,
                "char_end": 17,
                "resolution_methods": ["python_substring"],
                "source_term_kind": "preferred",
            }
        ],
    )
    prompt = Translator(settings, llm).build_prompt(
        "මුදල් අමාත්‍යාංශය",
        ContextBlock(text='{"entities":[{"source":"මුදල් අමාත්‍යාංශය","target":"wrong"}]}'),
        term,
    )

    assert "நிதி அமைச்சு" in prompt
    assert "take\nprecedence over conflicting renderings in discourse memory" in prompt
    assert "Grammatical inflection" in prompt


def test_all_terminology_nodes_finish_before_baseline_translation(settings):
    settings.translation.condition = "graft_baseline"
    settings.terminology.enabled = True
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=EchoLLM())
    app = dag_translate._build_memory_app(graph(), settings, llm).compile()
    events = list(
        app.stream(
            {"records": {}, "memories": {}, "memory_contexts": {}, "terminology": {}},
            stream_mode="updates",
        )
    )
    names = [next(iter(event)) for event in events]

    assert set(names[:2]) == {"terminology_0", "terminology_1"}
    assert names[2:] == ["translate_0", "memory_0", "translate_1", "memory_1"]
    assert [call.purpose for call in llm.ledger.calls[:2]] == [
        "terminology_extraction",
        "terminology_extraction",
    ]


def test_completed_terminology_nodes_resume_after_parallel_failure(settings, tmp_path):
    settings.translation.condition = "graft_baseline"
    settings.terminology.enabled = True
    settings.terminology.retry_on_parse_failure = 0
    settings.llm.use_cache = False
    echo = EchoLLM()
    failed = False
    term_calls: list[str] = []

    def responder(messages, params):
        nonlocal failed
        prompt = messages[0].content
        if prompt.startswith("You identify domain-specific terminology"):
            sid = "s1" if "අනුමත දීමනාව" in prompt else "s0"
            term_calls.append(sid)
            if sid == "s1" and not failed:
                failed = True
                raise RuntimeError("simulated terminology failure")
        return echo.complete(messages, **params).text

    llm = build_llm(settings, provider=FakeLLM(responder=responder))
    path = str(tmp_path / "terminology-checkpoints.sqlite")
    config = {"configurable": {"thread_id": "terminology-resume"}}
    initial = {"records": {}, "memories": {}, "memory_contexts": {}, "terminology": {}}
    with SqliteSaver.from_conn_string(path) as saver:
        app = dag_translate._build_memory_app(graph(), settings, llm).compile(checkpointer=saver)
        with pytest.raises(RuntimeError, match="simulated terminology failure"):
            app.invoke(initial, config)
    with SqliteSaver.from_conn_string(path) as saver:
        app = dag_translate._build_memory_app(graph(), settings, llm).compile(checkpointer=saver)
        result = app.invoke(None, config)

    assert set(result["terminology"]) == {"s0", "s1"}
    assert term_calls.count("s0") == 1
    assert term_calls.count("s1") == 2


def test_raw_context_overlay_disables_terminology():
    settings = load_settings(overlays=["configs/experiments/raw_context.yaml"])
    assert settings.translation.condition == "dag_raw_context"
    assert settings.terminology.enabled is False

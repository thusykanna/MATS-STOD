"""Text normalisation, and the zero-width joiner surviving the whole pipeline."""

from __future__ import annotations

import unicodedata

from langgraph.checkpoint.sqlite import SqliteSaver

from mats_stod.io.parallel import load_parallel
from mats_stod.io.text import ZWJ, normalise
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import EchoLLM, FakeLLM
from mats_stod.parsing.plaintext import PlainTextParser
from mats_stod.pipelines.dag_translate import translate_document
from mats_stod.schemas import DiscourseGraph
from mats_stod.segmentation.graft_discourse import GraftDiscourseSegmenter

# ශ්‍රී: the ZWJ between ් and ර is what makes this a conjunct.
SRI = "ශ්" + ZWJ + "රී"


def test_normalise_is_nfc_and_keeps_zwj():
    text = f"{SRI} ලංකා"
    out = normalise(text)
    assert out == unicodedata.normalize("NFC", out)
    assert ZWJ in out
    assert out.count(ZWJ) == 1


def test_normalise_strips_bom_and_normalises_line_endings():
    assert normalise("﻿a\r\nb\rc") == "a\nb\nc"


def test_sample_file_contains_zwj(sample_text):
    assert ZWJ in sample_text


def test_zwj_survives_parse_and_segment(sample_text, settings):
    doc = PlainTextParser().parse(sample_text, "circular_01", "si")
    settings.llm.use_cache = False
    llm = build_llm(settings, provider=FakeLLM(default="no"))
    result = GraftDiscourseSegmenter(settings, llm).segment(doc)
    joined = "".join(s.text for s in result.segments)
    assert joined.count(ZWJ) == sample_text.count(ZWJ)
    DiscourseGraph(doc_id=doc.doc_id, segments=result.segments).validate_graph(
        raw_text=doc.raw_text
    )


def test_zwj_survives_the_whole_translation_pipeline(settings, tmp_path):
    """A ZWJ present in the source is present in the output document.

    EchoLLM answers every segmentation/edge decision "no" and echoes the
    source back for translation, so any loss would be the DAG pipeline's
    doing (segmentation, graph assembly, context masking, reassembly)
    rather than the model's.
    """
    settings.llm.use_cache = False
    pair = load_parallel("data/samples", "si", "ta", doc_ids=["circular_01"])[0]
    llm = build_llm(settings, provider=EchoLLM())
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as checkpointer:
        result = translate_document(pair, settings, llm, checkpointer)
    assert result.output_text.count(ZWJ) == pair.source_text.count(ZWJ)
    assert ZWJ in result.output_text

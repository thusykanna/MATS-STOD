"""Text normalisation, and the zero-width joiner surviving the whole pipeline."""

from __future__ import annotations

import json
import unicodedata

from mats_stod.io.parallel import load_parallel
from mats_stod.io.text import ZWJ, normalise
from mats_stod.llm.factory import build_llm
from mats_stod.llm.fake import FakeLLM
from mats_stod.parsing.plaintext import PlainTextParser
from mats_stod.pipelines.baseline import translate_document
from mats_stod.schemas import DiscourseGraph
from mats_stod.segmentation.structural import StructuralSegmenter

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
    result = StructuralSegmenter(settings.segmentation).segment(doc)
    joined = "".join(s.text for s in result.segments)
    assert joined.count(ZWJ) == sample_text.count(ZWJ)
    DiscourseGraph(doc_id=doc.doc_id, segments=result.segments).validate_graph(
        raw_text=doc.raw_text
    )


def test_zwj_survives_the_whole_translation_pipeline(settings, tmp_path):
    """A ZWJ present in the source is present in the output document.

    The fake model echoes the source, so any loss would be the pipeline's
    doing rather than the model's.
    """

    def echo(messages, params):
        prompt = messages[-1].content
        body = prompt.split("Source text", 1)[1].split(":", 1)[1]
        body = body.split("Return a JSON", 1)[0].strip()
        return json.dumps({"translation": body}, ensure_ascii=False)

    settings.llm.use_cache = False
    settings.translation.strategy = "B1_isolated"
    pair = load_parallel("data/samples", "si", "ta", doc_ids=["circular_01"])[0]
    llm = build_llm(settings, provider=FakeLLM(responder=echo))
    result = translate_document(pair, settings, llm)
    assert result.output_text.count(ZWJ) == pair.source_text.count(ZWJ)
    assert ZWJ in result.output_text

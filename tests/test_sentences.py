"""Sentence splitting: the cases that break naive regex splitters."""

from __future__ import annotations

import pytest

from mats_stod.config import SentenceSplitSettings
from mats_stod.segmentation.sentences import SentenceSplitter


@pytest.fixture
def splitter(settings) -> SentenceSplitter:
    return SentenceSplitter(settings.segmentation.sentences)


def texts(splitter, s: str) -> list[str]:
    return [s[x.char_start : x.char_end] for x in splitter.split(s)]


def test_two_plain_sentences(splitter):
    s = "මෙම චක්‍රලේඛය අදාළ වේ. එය ක්‍රියාත්මක වේ."
    assert len(texts(splitter, s)) == 2


def test_decimal_number_does_not_split(splitter):
    s = "දුර 115.5 කි."
    assert texts(splitter, s) == [s]


def test_thousands_separator_does_not_split(splitter):
    s = "දීමනාව රු. 3,500.00 වේ."
    assert texts(splitter, s) == [s]


def test_sinhala_abbreviation_does_not_split(splitter):
    s = "දුර කි.මී. 115 කි."
    assert texts(splitter, s) == [s]


def test_tamil_abbreviation_does_not_split(splitter):
    s = "தூரம் கி.மீ. 115 ஆகும்."
    assert texts(splitter, s) == [s]


def test_abbreviation_prefix_that_is_also_a_word_still_ends_a_sentence(splitter):
    """"කි." opens "කි.මී." but is a complete word too; only the real
    abbreviation may suppress the boundary."""
    s = "දුර කි.මී. 115.5 කි. එම ගමන අවසන් විය."
    parts = texts(splitter, s)
    assert len(parts) == 2
    assert parts[0].endswith("කි.")


def test_clause_numbering_does_not_split(splitter):
    s = "2.2 ඉහත සඳහන් ලේඛන ඉදිරිපත් කළ යුතුය:"
    assert texts(splitter, s) == [s]


def test_clause_reference_inside_a_sentence_does_not_split(splitter):
    s = "දීමනාව 3.1 වගන්තිය යටතේ ගණනය කෙරේ."
    assert texts(splitter, s) == [s]


def test_list_items_split_on_line_breaks(splitter):
    s = "- පළමු\n- දෙවන\n- තෙවන"
    assert len(texts(splitter, s)) == 3


def test_offsets_index_the_original_string(splitter):
    s = "පළමු වාක්‍යය. දෙවන වාක්‍යය."
    for span in splitter.split(s):
        assert s[span.char_start : span.char_end] == span.text_of(s)


def test_offset_argument_shifts_spans(splitter):
    s = "abc. def."
    shifted = splitter.split(s, offset=100)
    assert shifted[0].char_start == 100


def test_question_and_exclamation_split():
    sp = SentenceSplitter(SentenceSplitSettings())
    s = "මෙය කුමක්ද? එය හොඳයි!"
    assert len(sp.split(s)) == 2


def test_empty_and_whitespace_input(splitter):
    assert splitter.split("") == []
    assert splitter.split("   \n  ") == []

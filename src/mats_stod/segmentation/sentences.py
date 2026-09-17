"""Sentence splitting for Sinhala and Tamil.

Both languages end sentences with the full stop, so the hard part is not
finding stops but ignoring the ones that end nothing: abbreviations (කි.මී.,
எ.கா.), decimals (115.5) and clause numbering (3.1, 2.2). Official documents
are full of all three, and a wrong split here propagates into every segment,
edge and translation downstream.

Splitting returns spans into the original string. Nothing is rewritten, so
offsets stay valid and invariant 1 cannot be broken by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import SentenceSplitSettings


@dataclass
class SentenceSpan:
    """One sentence as a span of the document text."""

    index: int
    char_start: int
    char_end: int

    def text_of(self, raw_text: str) -> str:
        return raw_text[self.char_start : self.char_end]


#: A digit immediately before the stop, as in 115.5 or 3,500.00
_DIGIT_BEFORE = re.compile(r"\d\s*$")
#: Clause numbering occupying its own line start: "2.2", "3.1.4"
_CLAUSE_NUMBER = re.compile(r"(?:\A|\n)[ \t]*\d+(?:\.\d+)*\Z")
#: A single-letter initial, as in "A." of "A. Perera"
_INITIAL = re.compile(r"(?:\A|[\s(])[^\W\d_]\Z", re.UNICODE)
#: Line that starts as a bullet or a numbered clause.
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*•·]|\(?\d+[.)]|\(?[a-z඀-෿஀-௿][.)])")


class SentenceSplitter:
    """Rule-based sentence splitter driven by config."""

    def __init__(self, settings: SentenceSplitSettings | None = None) -> None:
        self.settings = settings or SentenceSplitSettings()
        self._terminators = set(self.settings.terminators)
        self._abbreviations = sorted(self.settings.abbreviations, key=len, reverse=True)

    # -- public -----------------------------------------------------------

    def split(self, text: str, offset: int = 0) -> list[SentenceSpan]:
        """Split `text` into sentence spans, shifted by `offset`.

        `offset` lets a caller split one block and still receive spans that
        index the whole document.
        """
        cuts = self._cut_points(text)
        spans: list[SentenceSpan] = []
        start = 0
        for cut in cuts:
            s, e = _trim(text, start, cut)
            if e > s:
                spans.append(SentenceSpan(len(spans), s + offset, e + offset))
            start = cut
        s, e = _trim(text, start, len(text))
        if e > s:
            spans.append(SentenceSpan(len(spans), s + offset, e + offset))
        return self._merge_tiny(spans, text, offset)

    # -- internals --------------------------------------------------------

    def _cut_points(self, text: str) -> list[int]:
        """Offsets in `text` where one sentence ends and the next begins."""
        cuts: list[int] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch in self._terminators and not self._is_false_stop(text, i):
                end = i + 1
                # Absorb closing punctuation that belongs to this sentence.
                while end < n and text[end] in '"”’)]}':
                    end += 1
                cuts.append(end)
                i = end
                continue
            if ch == "\n" and self._breaks_at_newline(text, i):
                cuts.append(i + 1)
            i += 1
        return cuts

    def _is_false_stop(self, text: str, i: int) -> bool:
        """True when the terminator at `i` does not end a sentence."""
        if text[i] != ".":
            # Only the full stop is ambiguous. ? and ! do not occur inside
            # abbreviations or numbers in these documents.
            return False

        before = text[:i]
        ending = text[: i + 1]
        after = text[i + 1 :]

        # Decimal or thousands separator: 115.5, 3,500.00, and clause refs 3.1
        if _DIGIT_BEFORE.search(before) and after[:1].isdigit():
            return True

        # Clause numbering standing alone on its line: "2.2 ..." at line start
        if _CLAUSE_NUMBER.search(before):
            return True

        # A known abbreviation ends here, at its final stop or an internal one.
        # An internal stop only counts when the rest of the abbreviation
        # actually follows: "කි." opens "කි.මී." but is also a complete word,
        # and without the lookahead every occurrence of it would swallow the
        # sentence boundary after it.
        for abbr in self._abbreviations:
            if ending.endswith(abbr):
                return True
            for cut in range(1, len(abbr)):
                if (
                    abbr[cut - 1] == "."
                    and ending.endswith(abbr[:cut])
                    and after.startswith(abbr[cut:])
                ):
                    return True

        # A single-letter initial: "A." in a name.
        if _INITIAL.search(before):
            return True

        # A stop glued to a following letter is internal punctuation.
        if after[:1] and not after[:1].isspace() and not after[:1].isdigit():
            return True

        return False

    def _breaks_at_newline(self, text: str, i: int) -> bool:
        """Whether a hard line break ends a sentence.

        Official documents put headings and list items on their own lines with
        no full stop. Without this rule such a line is glued to whatever
        follows. The length guard keeps a hard-wrapped running paragraph from
        being chopped at every line.
        """
        line_start = text.rfind("\n", 0, i) + 1
        line = text[line_start:i].strip()
        if not line:
            return False
        if line[-1] in self._terminators:
            return True
        next_line = text[i + 1 :].split("\n", 1)[0]
        if _LIST_MARKER.match(line) or _LIST_MARKER.match(next_line):
            return True
        return len(line) <= self.settings.newline_break_max_chars

    def _merge_tiny(
        self, spans: list[SentenceSpan], text: str, offset: int
    ) -> list[SentenceSpan]:
        """Fold away fragments too short to be sentences.

        A stray fragment joins the previous sentence rather than standing
        alone, so the GRAFT segmenter never spends an LLM call on noise.
        """
        if not spans:
            return spans
        out: list[SentenceSpan] = []
        for span in spans:
            body = text[span.char_start - offset : span.char_end - offset]
            if out and len(body.strip()) < self.settings.min_sentence_chars:
                prev = out[-1]
                out[-1] = SentenceSpan(prev.index, prev.char_start, span.char_end)
                continue
            out.append(SentenceSpan(len(out), span.char_start, span.char_end))
        return out


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end

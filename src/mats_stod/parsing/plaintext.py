"""Plain-text parser: blank-line paragraphs only.

Layout detection (headings, numbered clauses, lists, tables) is future work by
an explicit project decision. Every block therefore comes out as a paragraph
at level 0 with an empty heading path. The consequence is recorded in the run
report: structural edges are empty until a real layout parser exists.
"""

from __future__ import annotations

import re

from ..schemas import Document, LayoutBlock
from .base import LayoutParser

#: One or more blank lines separate paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class PlainTextParser(LayoutParser):
    name = "plaintext"

    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        blocks: list[LayoutBlock] = []
        for index, (start, end) in enumerate(_paragraph_spans(text)):
            blocks.append(
                LayoutBlock(
                    block_id=f"{doc_id}-b{index:04d}",
                    type="paragraph",
                    level=0,
                    char_start=start,
                    char_end=end,
                )
            )
        return Document(
            doc_id=doc_id,
            source_lang=source_lang,
            raw_text=text,
            blocks=blocks,
            metadata={"parser": self.name, "layout_detected": False, **metadata},
        )


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Spans of non-empty paragraphs, trimmed of surrounding whitespace.

    Whitespace between paragraphs belongs to no block. Invariant 1 tolerates
    exactly that: dropped inter-segment whitespace, never dropped characters.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        chunk = text[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        tail = len(chunk) - len(chunk.rstrip())
        s, e = start + lead, end - tail
        if e > s:
            trimmed.append((s, e))
    return trimmed

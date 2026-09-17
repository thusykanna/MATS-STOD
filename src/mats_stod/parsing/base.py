"""Layout parsing interface.

Layout extraction is deliberately deferred: only `PlainTextParser` is
implemented, so every block is a paragraph and no heading structure exists
yet. The interface is fixed now so that a Markdown, DOCX or PDF parser can be
added later without changing segmentation, graph or translation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Document


class LayoutParser(ABC):
    """Turns raw text into a `Document` with layout blocks."""

    name: str = "base"

    @abstractmethod
    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        """Parse normalised text into a Document.

        `text` must already be NFC-normalised by the loader; parsers never
        modify characters, they only record spans, so that every offset stays
        valid against the same string.
        """


def build_parser(name: str) -> LayoutParser:
    if name in {"plaintext", "plain", "txt"}:
        from .plaintext import PlainTextParser

        return PlainTextParser()
    if name in {"markdown", "md"}:
        from .markdown import MarkdownParser

        return MarkdownParser()
    if name == "pdf":
        from .pdf import PdfParser

        return PdfParser()
    if name == "docx":
        from .docx import DocxParser

        return DocxParser()
    raise ValueError(f"unknown parser: {name!r}")

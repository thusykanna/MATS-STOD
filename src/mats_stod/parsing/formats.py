"""Markdown, PDF and DOCX layout parsers: future work.

All three are deferred by project decision (same interface, richer block
types to come). Implementing one means populating `blocks` with heading
levels, list items and table cells, which is what makes structural edges
non-empty.
"""

from __future__ import annotations

from ..schemas import Document
from .base import LayoutParser


class MarkdownParser(LayoutParser):
    name = "markdown"

    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        raise NotImplementedError(
            "MarkdownParser is future work; layout extraction is deferred (use plaintext)."
        )


class PdfParser(LayoutParser):
    name = "pdf"

    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        raise NotImplementedError("PdfParser is future work; convert to .txt first.")


class DocxParser(LayoutParser):
    name = "docx"

    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        raise NotImplementedError("DocxParser is future work; convert to .txt first.")

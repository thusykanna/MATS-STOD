"""Markdown layout parser: future work.

Deferred with PDF and DOCX by project decision. Implementing it means
populating `blocks` with heading levels, list items and table cells, which is
what makes structural edges non-empty.
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

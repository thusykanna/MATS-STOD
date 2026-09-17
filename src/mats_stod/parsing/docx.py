"""DOCX layout parser: future work (same interface, richer block types)."""

from __future__ import annotations

from ..schemas import Document
from .base import LayoutParser


class DocxParser(LayoutParser):
    name = "docx"

    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        raise NotImplementedError("DocxParser is future work; convert to .txt first.")

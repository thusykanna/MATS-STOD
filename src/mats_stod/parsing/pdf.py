"""PDF layout parser: future work (needs a layout model and OCR, out of scope)."""

from __future__ import annotations

from ..schemas import Document
from .base import LayoutParser


class PdfParser(LayoutParser):
    name = "pdf"

    def parse(self, text: str, doc_id: str, source_lang: str, **metadata: object) -> Document:
        raise NotImplementedError("PdfParser is future work; convert to .txt first.")

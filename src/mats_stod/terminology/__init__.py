"""Terminology extraction and local bilingual glossary lookup."""

from .agent import TerminologyAgent, TerminologyExtractionError
from .glossary import GlossaryStore, lookup_glossary
from .models import TerminologyRecord

__all__ = [
    "GlossaryStore",
    "TerminologyAgent",
    "TerminologyExtractionError",
    "TerminologyRecord",
    "lookup_glossary",
]

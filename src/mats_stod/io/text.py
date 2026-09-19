"""Text loading and normalisation.

Everything entering the system passes through `normalise` exactly once, at
load time. Downstream code never normalises again, because a second pass would
shift character offsets that segments and edges already point at.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

#: Zero-width joiner. In Sinhala it forms conjunct letters (for example in
#: "ශ්‍රී"); removing it produces a different, wrong word. NFC leaves it in
#: place, so the only real risk is well-meaning cleanup code.
ZWJ = "‍"
ZWNJ = "‌"


def normalise(text: str) -> str:
    """NFC-normalise, unify line endings, strip a BOM.

    NFC is chosen over NFD because Sinhala and Tamil combining marks compare
    and score differently when decomposed, and chrF++ is character-based.
    """
    text = text.replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def read_text(path: str | Path) -> str:
    """Read a UTF-8 file and normalise it."""
    return normalise(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def count_zwj(text: str) -> int:
    return text.count(ZWJ)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Rough token estimate used only for budgeting decisions.

    Reported token counts always come from the provider; this is for deciding
    how much discourse-graph context (`translation.context.build_dag_context`)
    fits before a segment is sent for translation.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return int(len(text) / chars_per_token) + 1

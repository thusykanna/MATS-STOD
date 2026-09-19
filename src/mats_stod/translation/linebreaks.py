"""Newline round-tripping through the translation call.

The model receives the segment's exact character sequence, newlines included,
as plain text inside a prompt. Nothing about "translate this" tells a model to
keep line breaks where they were, and asking it to return JSON only makes that
worse: a model asked for a JSON string routinely reflows the value into one
line rather than emitting escaped `\\n`s. Line breaks are masked as an opaque
token before the call and restored after, so fidelity does not depend on the
model choosing to comply.
"""

from __future__ import annotations

import re

#: Visually inert, JSON-safe (no quote, backslash or control character) and
#: vanishingly unlikely to occur in a Sinhala/Tamil government document.
LINEBREAK_TOKEN = "⟦¶⟧"

_TOKEN_WITH_SPACE = re.compile(rf"\s*{re.escape(LINEBREAK_TOKEN)}\s*")


def mask_linebreaks(text: str) -> str:
    """Replace every newline with the line-break token."""
    return text.replace("\n", LINEBREAK_TOKEN)


def unmask_linebreaks(text: str) -> str:
    """Replace every line-break token with a newline.

    Whitespace a model adds around the token (a space before or after is
    common) is absorbed into the same substitution rather than left behind.
    """
    return _TOKEN_WITH_SPACE.sub("\n", text)

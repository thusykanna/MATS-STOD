"""Protected content checks.

Official documents carry reference numbers, dates and amounts where a single
changed digit makes the translation wrong in a way no fluency metric detects.
This module flags such divergences. It deliberately does not repair them:
a silent repair would hide how often the model gets them wrong, which is
exactly the number worth reporting.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..config import ProtectSettings

#: Digits used in Sinhala and Tamil documents are the Western forms, but the
#: check normalises any Unicode decimal digit so a locale digit is comparable.
_DIGIT = re.compile(r"\d")


@dataclass
class ProtectionReport:
    ok: bool
    flags: list[str]
    source_digits: list[str]
    target_digits: list[str]
    missing_patterns: list[str]


def _digits(text: str) -> list[str]:
    return _DIGIT.findall(text)


def _digit_runs(text: str) -> list[str]:
    """Contiguous digit runs, which is what a reference number really is."""
    return re.findall(r"\d+", text)


def check(source: str, target: str, settings: ProtectSettings) -> ProtectionReport:
    """Compare protected content between a source segment and its translation."""
    flags: list[str] = []
    if not settings.enabled:
        return ProtectionReport(True, [], [], [], [])

    src_digits = _digits(source)
    tgt_digits = _digits(target)

    if settings.check_digits and Counter(src_digits) != Counter(tgt_digits):
        flags.append(
            f"digit_mismatch: source has {len(src_digits)} digits, target has {len(tgt_digits)}"
        )

    src_runs = Counter(_digit_runs(source))
    tgt_runs = Counter(_digit_runs(target))
    dropped = sorted((src_runs - tgt_runs).elements())
    if dropped:
        flags.append(f"numbers_missing_in_target: {dropped[:10]}")

    missing_patterns: list[str] = []
    for pattern in settings.patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            flags.append(f"bad_protect_pattern: {pattern!r} ({exc})")
            continue
        for match in compiled.findall(source):
            value = match if isinstance(match, str) else match[0]
            if value and value not in target:
                missing_patterns.append(value)
    if missing_patterns:
        unique = sorted(set(missing_patterns))
        flags.append(f"protected_pattern_missing: {unique[:10]}")

    return ProtectionReport(
        ok=not flags,
        flags=flags,
        source_digits=src_digits,
        target_digits=tgt_digits,
        missing_patterns=sorted(set(missing_patterns)),
    )

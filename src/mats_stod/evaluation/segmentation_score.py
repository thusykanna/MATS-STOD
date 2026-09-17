"""Segmentation scoring against gold boundaries.

Boundary F1 answers "did it put boundaries in the right places", but it
punishes a near miss as hard as a wild guess. Pk and WindowDiff answer "how
badly wrong is it" by sliding a window over the document, so a boundary placed
one sentence late costs little. Both are reported because they disagree in
informative ways.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ..schemas import Segment


@dataclass
class SegmentationScore:
    doc_id: str
    precision: float
    recall: float
    f1: float
    pk: float
    window_diff: float
    n_gold_boundaries: int
    n_pred_boundaries: int
    window_size: int


def boundaries_from_segments(segments: list[Segment]) -> list[int]:
    """Internal boundaries as character offsets.

    The start of the first segment is not a boundary: every segmentation has
    it, so counting it inflates agreement.
    """
    ordered = sorted(segments, key=lambda s: s.order)
    return [s.char_start for s in ordered[1:]]


def boundary_prf(
    gold: list[int], predicted: list[int], tolerance: int = 0
) -> tuple[float, float, float]:
    """Precision, recall and F1 over boundary offsets, within a tolerance.

    A tolerance is needed because gold boundaries are annotated by hand at the
    start of a sentence while a segmenter may place them after the preceding
    whitespace; those are the same decision.
    """
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    unmatched = list(gold)
    hits = 0
    for p in predicted:
        for i, g in enumerate(unmatched):
            if abs(p - g) <= tolerance:
                hits += 1
                unmatched.pop(i)
                break
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _to_mass(boundaries: list[int], length: int, unit: int) -> list[int]:
    """Turn character boundaries into a 0/1 sequence over fixed-size units.

    Pk and WindowDiff are defined over discrete units. Characters are used as
    the unit, sub-sampled by `unit`, because sentence indices are not
    available for a segmentation that disagrees with gold about sentences.
    """
    n = max(1, length // unit)
    marks = [0] * n
    for b in boundaries:
        idx = min(n - 1, max(0, b // unit))
        marks[idx] = 1
    return marks


def _window_size(gold_marks: list[int]) -> int:
    """Half the mean gold segment length, the standard choice for Pk."""
    n_bounds = sum(gold_marks)
    if n_bounds == 0:
        return max(2, len(gold_marks) // 2)
    mean_seg = len(gold_marks) / (n_bounds + 1)
    return max(2, int(round(mean_seg / 2)))


def pk_and_windowdiff(
    gold: list[int], predicted: list[int], length: int, unit: int = 20
) -> tuple[float, float, int]:
    """Compute Pk and WindowDiff over character units."""
    gold_marks = _to_mass(gold, length, unit)
    pred_marks = _to_mass(predicted, length, unit)
    n = len(gold_marks)
    k = _window_size(gold_marks)
    if n <= k:
        return 0.0, 0.0, k

    pk_errors = 0
    wd_errors = 0
    windows = 0
    for i in range(n - k):
        g_same = sum(gold_marks[i + 1 : i + k + 1]) == 0
        p_same = sum(pred_marks[i + 1 : i + k + 1]) == 0
        if g_same != p_same:
            pk_errors += 1
        g_count = sum(gold_marks[i + 1 : i + k + 1])
        p_count = sum(pred_marks[i + 1 : i + k + 1])
        if g_count != p_count:
            wd_errors += 1
        windows += 1
    return (
        round(pk_errors / windows, 4),
        round(wd_errors / windows, 4),
        k,
    )


def score_segmentation(
    doc_id: str,
    predicted_segments: list[Segment],
    gold_boundaries: list[int],
    doc_length: int,
    tolerance: int = 2,
    unit: int = 20,
) -> SegmentationScore:
    predicted = boundaries_from_segments(predicted_segments)
    p, r, f1 = boundary_prf(gold_boundaries, predicted, tolerance=tolerance)
    pk, wd, k = pk_and_windowdiff(gold_boundaries, predicted, doc_length, unit=unit)
    return SegmentationScore(
        doc_id=doc_id,
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f1, 4),
        pk=pk,
        window_diff=wd,
        n_gold_boundaries=len(gold_boundaries),
        n_pred_boundaries=len(predicted),
        window_size=k,
    )


def aggregate(scores: list[SegmentationScore]) -> dict[str, Any]:
    if not scores:
        return {}
    return {
        "precision": round(mean(s.precision for s in scores), 4),
        "recall": round(mean(s.recall for s in scores), 4),
        "f1": round(mean(s.f1 for s in scores), 4),
        "pk": round(mean(s.pk for s in scores), 4),
        "window_diff": round(mean(s.window_diff for s in scores), 4),
        "n_docs": len(scores),
    }


def load_gold_boundaries(path: str | Path) -> list[int]:
    """Read a gold segmentation file.

    The file is JSON: {"doc_id": ..., "boundaries": [char offsets]}.
    """
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return sorted(int(x) for x in data)
    return sorted(int(x) for x in data.get("boundaries", []))


def build_report_section(scores: list[SegmentationScore]) -> dict[str, Any]:
    return {"documents": [asdict(s) for s in scores], "corpus": aggregate(scores)}

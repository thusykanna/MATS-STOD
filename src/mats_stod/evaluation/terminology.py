"""Gold annotations and metrics for terminology discovery and resolution."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import REPO_ROOT
from ..schemas import Segment, TranslationRecord
from ..terminology.glossary import GlossaryStore
from ..terminology.models import TerminologyRecord


class GoldTerminologyItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    entry_id: str | None = None
    accepted_target_forms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_span(self) -> GoldTerminologyItem:
        if self.char_end <= self.char_start:
            raise ValueError("gold terminology span must be non-empty")
        return self


class GoldTerminologyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    language: str
    terms: list[GoldTerminologyItem]


def load_terminology_gold(
    root: str | Path, doc_id: str, language: str, source_text: str
) -> GoldTerminologyDocument | None:
    path = Path(root)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path / f"{doc_id}.{language}.json"
    if not path.exists():
        return None
    gold = GoldTerminologyDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if gold.doc_id != doc_id or gold.language != language:
        raise ValueError(f"terminology gold identity mismatch in {path}")
    for term in gold.terms:
        if source_text[term.char_start:term.char_end] != term.surface:
            raise ValueError(f"terminology gold span does not match source in {path}: {term}")
    return gold


def _prf(tp: int, predicted: int, gold: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score_terminology(
    gold: GoldTerminologyDocument,
    source_text: str,
    segments: list[Segment],
    records: list[TerminologyRecord],
    translations: list[TranslationRecord],
    glossary: GlossaryStore,
    source_lang: str,
    target_lang: str,
) -> dict[str, object]:
    segments_by_id = {segment.seg_id: segment for segment in segments}
    translations_by_id = {record.seg_id: record for record in translations}
    discovered: set[tuple[int, int]] = set()
    resolved: list[tuple[int, int, str, tuple[str, ...], str]] = []

    for record in records:
        segment = segments_by_id[record.seg_id]
        for candidate in record.candidates:
            start = 0
            while (at := segment.text.find(candidate.surface, start)) >= 0:
                discovered.add(
                    (segment.char_start + at, segment.char_start + at + len(candidate.surface))
                )
                start = at + len(candidate.surface)
        for match in record.matches:
            start = segment.char_start + match.char_start
            end = segment.char_start + match.char_end
            discovered.add((start, end))
            resolved.append(
                (start, end, match.entry_id, tuple(match.resolution_methods), record.seg_id)
            )

    gold_spans = {(term.char_start, term.char_end) for term in gold.terms}
    identification_tp = len(discovered & gold_spans)
    identification = {
        "tp": identification_tp,
        "predicted": len(discovered),
        "gold": len(gold_spans),
        **_prf(identification_tp, len(discovered), len(gold_spans)),
    }

    gold_resolved = {
        (term.char_start, term.char_end, term.entry_id)
        for term in gold.terms
        if term.entry_id is not None
    }
    predicted_resolved = {(start, end, entry) for start, end, entry, _, _ in resolved}
    correct_resolutions = predicted_resolved & gold_resolved
    resolution = {
        "correct": len(correct_resolutions),
        "predicted": len(predicted_resolved),
        "gold": len(gold_resolved),
        "false_mappings": len(predicted_resolved - gold_resolved),
        **_prf(len(correct_resolutions), len(predicted_resolved), len(gold_resolved)),
    }

    recoverable = {
        (term.char_start, term.char_end, term.entry_id)
        for term in gold.terms
        if term.entry_id is not None
        and glossary.lookup(term.surface, source_lang, target_lang, glossary.data.domain) is None
    }
    recovered = {
        (start, end, entry)
        for start, end, entry, methods, _ in resolved
        if "llm_surface_exact" not in methods
    } & recoverable

    method_counts: dict[str, int] = {}
    for _, _, _, methods, _ in resolved:
        key = "+".join(methods)
        method_counts[key] = method_counts.get(key, 0) + 1

    target_eligible = target_correct = 0
    gold_by_resolution = {
        (term.char_start, term.char_end, term.entry_id): term
        for term in gold.terms
        if term.entry_id is not None
    }
    for start, end, entry, _, seg_id in resolved:
        term = gold_by_resolution.get((start, end, entry))
        if term is None or not term.accepted_target_forms:
            continue
        target_eligible += 1
        target_text = unicodedata.normalize("NFC", translations_by_id[seg_id].target_text)
        accepted = [unicodedata.normalize("NFC", form) for form in term.accepted_target_forms]
        target_correct += int(any(form in target_text for form in accepted))

    return {
        "doc_id": gold.doc_id,
        "language": gold.language,
        "identification": identification,
        "resolution": resolution,
        "recovery": {
            "correct": len(recovered),
            "eligible": len(recoverable),
            "recall": round(len(recovered) / len(recoverable), 4) if recoverable else 0.0,
        },
        "target_realization": {
            "correct": target_correct,
            "eligible": target_eligible,
            "accuracy": round(target_correct / target_eligible, 4) if target_eligible else 0.0,
        },
        "resolution_methods": dict(sorted(method_counts.items())),
        "source_chars": len(source_text),
    }


def aggregate_terminology_scores(scores: list[dict[str, object]]) -> dict[str, object] | None:
    if not scores:
        return None
    id_tp = sum(int(score["identification"]["tp"]) for score in scores)  # type: ignore[index]
    id_pred = sum(int(score["identification"]["predicted"]) for score in scores)  # type: ignore[index]
    id_gold = sum(int(score["identification"]["gold"]) for score in scores)  # type: ignore[index]
    res_correct = sum(int(score["resolution"]["correct"]) for score in scores)  # type: ignore[index]
    res_pred = sum(int(score["resolution"]["predicted"]) for score in scores)  # type: ignore[index]
    res_gold = sum(int(score["resolution"]["gold"]) for score in scores)  # type: ignore[index]
    recovery_correct = sum(int(score["recovery"]["correct"]) for score in scores)  # type: ignore[index]
    recovery_eligible = sum(int(score["recovery"]["eligible"]) for score in scores)  # type: ignore[index]
    target_correct = sum(int(score["target_realization"]["correct"]) for score in scores)  # type: ignore[index]
    target_eligible = sum(int(score["target_realization"]["eligible"]) for score in scores)  # type: ignore[index]
    false_mappings = sum(int(score["resolution"]["false_mappings"]) for score in scores)  # type: ignore[index]
    return {
        "documents": len(scores),
        "identification": {"tp": id_tp, "predicted": id_pred, "gold": id_gold,
                           **_prf(id_tp, id_pred, id_gold)},
        "resolution": {"correct": res_correct, "predicted": res_pred, "gold": res_gold,
                       "false_mappings": false_mappings,
                       **_prf(res_correct, res_pred, res_gold)},
        "recovery": {"correct": recovery_correct, "eligible": recovery_eligible,
                     "recall": round(recovery_correct / recovery_eligible, 4)
                     if recovery_eligible else 0.0},
        "target_realization": {"correct": target_correct, "eligible": target_eligible,
                               "accuracy": round(target_correct / target_eligible, 4)
                               if target_eligible else 0.0},
    }

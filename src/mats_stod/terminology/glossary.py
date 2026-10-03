"""Deterministic local glossary loading, lookup, and source-text scanning."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from ..config import REPO_ROOT, Settings
from .models import GlossaryData, GlossaryEntry, GlossaryLookupResult, TerminologyMatch


def normalize_term(value: str) -> str:
    """Stable lookup normalization without stemming or fuzzy rewriting."""
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


class GlossaryStore:
    def __init__(self, data: GlossaryData, content_hash: str) -> None:
        self.data = data
        self.content_hash = content_hash
        self._index: dict[str, dict[str, tuple[GlossaryEntry, str, str]]] = {
            "si": {},
            "ta": {},
        }
        for entry in data.entries:
            for lang, localized in entry.terms.items():
                for kind, forms in (
                    ("preferred", [localized.preferred]),
                    ("alias", localized.aliases),
                ):
                    for form in forms:
                        key = normalize_term(form)
                        existing = self._index[lang].get(key)
                        if existing is not None:
                            if existing[0].id == entry.id:
                                raise ValueError(
                                    f"duplicate glossary key {form!r} in {entry.id} for {lang}"
                                )
                            raise ValueError(
                                f"ambiguous glossary key {form!r} for {lang}: "
                                f"{existing[0].id} and {entry.id}"
                            )
                        self._index[lang][key] = (entry, kind, form)

    @classmethod
    def load(cls, path: str | Path) -> GlossaryStore:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        raw = resolved.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid glossary JSON at {resolved}: {exc}") from exc
        return cls(GlossaryData.model_validate(payload), hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_settings(cls, settings: Settings) -> GlossaryStore:
        return cls.load(settings.terminology.glossary_path)

    def lookup(
        self,
        query: str,
        source_lang: str,
        target_lang: str,
        domain: str | None = None,
    ) -> GlossaryLookupResult | None:
        self.validate_direction(source_lang, target_lang)
        if domain is not None and normalize_term(domain) != normalize_term(self.data.domain):
            return None
        found = self._index[source_lang].get(normalize_term(query))
        if found is None:
            return None
        entry, kind, matched_source = found
        return GlossaryLookupResult(
            entry_id=entry.id,
            definition=entry.definition,
            source_preferred=entry.terms[source_lang].preferred,
            target_preferred=entry.terms[target_lang].preferred,
            matched_source=matched_source,
            source_term_kind=kind,
        )

    def find_in_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> list[TerminologyMatch]:
        """Find literal glossary forms, preferring the longest overlapping match."""
        self.validate_direction(source_lang, target_lang)
        # Input files are NFC-normalized at load time. Search the original
        # source: case folding can expand characters and invalidate offsets.
        proposed: list[TerminologyMatch] = []
        for entry in self.data.entries:
            localized = entry.terms[source_lang]
            for kind, forms in (
                ("preferred", [localized.preferred]),
                ("alias", localized.aliases),
            ):
                for form in forms:
                    needle = unicodedata.normalize("NFC", form)
                    start = 0
                    while needle and (at := text.find(needle, start)) >= 0:
                        end = at + len(needle)
                        proposed.append(
                            TerminologyMatch(
                                entry_id=entry.id,
                                definition=entry.definition,
                                source_surface=text[at:end],
                                lookup_form=form,
                                source_preferred=localized.preferred,
                                target_preferred=entry.terms[target_lang].preferred,
                                char_start=at,
                                char_end=end,
                                resolution_methods=["python_substring"],
                                source_term_kind=kind,
                            )
                        )
                        start = end
        return _longest_non_overlapping(proposed)

    def snapshot(self) -> dict[str, object]:
        return {
            **self.data.model_dump(mode="json"),
            "sha256": self.content_hash,
        }

    def validate_direction(self, source_lang: str, target_lang: str) -> None:
        supported = set(self._index)
        if source_lang not in supported or target_lang not in supported:
            raise ValueError(
                f"glossary {self.data.version!r} supports {sorted(supported)}, "
                f"not {source_lang!r}->{target_lang!r}"
            )
        if source_lang == target_lang:
            raise ValueError("glossary source and target languages must differ")


def lookup_glossary(
    store: GlossaryStore,
    query: str,
    source_lang: str,
    target_lang: str,
    domain: str | None = None,
) -> GlossaryLookupResult | None:
    """Provider-neutral lookup tool used for agent-produced queries."""
    return store.lookup(query, source_lang, target_lang, domain)


def _longest_non_overlapping(matches: list[TerminologyMatch]) -> list[TerminologyMatch]:
    merged: dict[tuple[str, int, int], TerminologyMatch] = {}
    for match in matches:
        key = (match.entry_id, match.char_start, match.char_end)
        existing = merged.get(key)
        if existing is None:
            merged[key] = match
            continue
        merged[key] = existing.model_copy(update={
            "resolution_methods": sorted(set(
                existing.resolution_methods + match.resolution_methods
            ))
        })
    ranked = sorted(
        merged.values(),
        key=lambda match: (
            -(match.char_end - match.char_start),
            match.char_start,
            match.entry_id,
            tuple(match.resolution_methods),
        ),
    )
    accepted: list[TerminologyMatch] = []
    for match in ranked:
        if any(
            match.char_start < existing.char_end and existing.char_start < match.char_end
            for existing in accepted
        ):
            continue
        accepted.append(match)
    return sorted(accepted, key=lambda match: (match.char_start, match.char_end, match.entry_id))

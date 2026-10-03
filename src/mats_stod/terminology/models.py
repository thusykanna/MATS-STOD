"""Validated glossary and per-segment terminology records."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class LocalizedTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred: Text
    aliases: list[Text] = Field(default_factory=list)

    @field_validator("preferred")
    @classmethod
    def normalize_preferred(cls, value: str) -> str:
        return unicodedata.normalize("NFC", value)

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, values: list[str]) -> list[str]:
        return [unicodedata.normalize("NFC", value) for value in values]


class GlossaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    definition: Text
    terms: dict[str, LocalizedTerm]

    @model_validator(mode="after")
    def require_sinhala_and_tamil(self) -> GlossaryEntry:
        if set(self.terms) != {"si", "ta"}:
            raise ValueError(f"entry {self.id}: terms must contain exactly si and ta")
        return self


class GlossaryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Text
    domain: Text
    notice: Text | None = None
    entries: list[GlossaryEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> GlossaryData:
        ids = [entry.id for entry in self.entries]
        duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate glossary entry IDs: {duplicates}")
        return self


class TerminologyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface: Text
    reason: Text


class CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[TerminologyCandidate]


class GlossaryLookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    definition: str
    source_preferred: str
    target_preferred: str
    matched_source: str
    source_term_kind: Literal["preferred", "alias"]


class TerminologyMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    definition: str
    source_surface: str
    source_preferred: str
    target_preferred: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    match_method: Literal["deterministic", "agent"]
    source_term_kind: Literal["preferred", "alias"]

    @model_validator(mode="after")
    def valid_span(self) -> TerminologyMatch:
        if self.char_end <= self.char_start:
            raise ValueError("terminology match must have a non-empty span")
        return self


class TerminologyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seg_id: str
    candidates: list[TerminologyCandidate] = Field(default_factory=list)
    matches: list[TerminologyMatch] = Field(default_factory=list)
    unmatched_candidates: list[str] = Field(default_factory=list)
    glossary_version: str
    glossary_hash: str
    model: str
    prompt_version: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    parse_failures: list[str] = Field(default_factory=list)
    prompt: str = ""

    def prompt_payload(self) -> list[dict[str, str]]:
        """Compact approved pairs supplied to the translation prompt."""
        seen: set[tuple[str, str, str]] = set()
        payload: list[dict[str, str]] = []
        for match in self.matches:
            key = (match.entry_id, match.source_surface, match.target_preferred)
            if key in seen:
                continue
            seen.add(key)
            payload.append(
                {
                    "entry_id": match.entry_id,
                    "source": match.source_surface,
                    "target_preferred": match.target_preferred,
                    "definition": match.definition,
                }
            )
        return payload

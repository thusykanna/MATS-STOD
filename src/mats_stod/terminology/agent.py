"""LLM term identification followed by deterministic local glossary lookup."""

from __future__ import annotations

import time

from pydantic import ValidationError

from ..config import Settings
from ..io.text import estimate_tokens
from ..llm.base import LLMError, Message, ParseError, parse_json_response
from ..llm.client import CachedLLM
from ..prompts.registry import render
from ..schemas import Segment
from .glossary import GlossaryStore, _longest_non_overlapping, lookup_glossary
from .models import (
    CandidateResponse,
    TerminologyCandidate,
    TerminologyMatch,
    TerminologyRecord,
    UnmatchedTerminologyCandidate,
)

TERMINOLOGY_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "lookup_form": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["surface", "lookup_form", "reason"],
            },
        }
    },
    "required": ["terms"],
}


class TerminologyExtractionError(LLMError):
    pass


class TerminologyAgent:
    def __init__(self, settings: Settings, llm: CachedLLM, glossary: GlossaryStore) -> None:
        self.settings = settings
        self.llm = llm
        self.glossary = glossary

    def extract(self, segment: Segment) -> TerminologyRecord:
        started = time.perf_counter()
        method = self.settings.terminology.method
        candidates: list[TerminologyCandidate] = []
        final_prompt = ""
        tokens_in = tokens_out = 0
        cached = True
        failures: list[str] = []

        if method != "python_scan":
            candidates, final_prompt, tokens_in, tokens_out, cached, failures = (
                self._extract_llm_candidates(segment)
            )

        matches = (
            self.glossary.find_in_text(
                segment.text, self.settings.langs.source, self.settings.langs.target
            )
            if method in {"python_scan", "hybrid"}
            else []
        )
        unmatched: list[UnmatchedTerminologyCandidate] = []
        if method != "python_scan":
            allow_lookup_form = method in {"llm_lookup_form", "hybrid"}
            for candidate in candidates:
                hit = lookup_glossary(
                    self.glossary,
                    candidate.surface,
                    self.settings.langs.source,
                    self.settings.langs.target,
                    self.glossary.data.domain,
                )
                resolution_method = "llm_surface_exact"
                if hit is None and allow_lookup_form:
                    hit = lookup_glossary(
                        self.glossary,
                        candidate.lookup_form,
                        self.settings.langs.source,
                        self.settings.langs.target,
                        self.glossary.data.domain,
                    )
                    resolution_method = "llm_lookup_form"
                if hit is None:
                    unmatched.append(UnmatchedTerminologyCandidate(
                        surface_form=candidate.surface,
                        lookup_form=candidate.lookup_form,
                        reason=candidate.reason,
                    ))
                    continue
                start = 0
                while (at := segment.text.find(candidate.surface, start)) >= 0:
                    end = at + len(candidate.surface)
                    matches.append(
                        TerminologyMatch(
                            entry_id=hit.entry_id,
                            definition=hit.definition,
                            source_surface=segment.text[at:end],
                            lookup_form=hit.matched_source,
                            source_preferred=hit.source_preferred,
                            target_preferred=hit.target_preferred,
                            char_start=at,
                            char_end=end,
                            resolution_methods=[resolution_method],
                            source_term_kind=hit.source_term_kind,
                        )
                    )
                    start = end

        return TerminologyRecord(
            seg_id=segment.seg_id,
            extraction_method=method,
            candidates=candidates,
            matches=_longest_non_overlapping(matches),
            unmatched_candidates=list({item.surface_form: item for item in unmatched}.values()),
            glossary_version=self.glossary.data.version,
            glossary_hash=self.glossary.content_hash,
            model=self.llm.model,
            prompt_version=self.settings.terminology.prompt_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=round(time.perf_counter() - started, 6),
            cached=cached,
            parse_failures=failures,
            prompt=final_prompt,
        )

    def _extract_llm_candidates(
        self, segment: Segment
    ) -> tuple[list[TerminologyCandidate], str, int, int, bool, list[str]]:
        prompt = render(
            self.settings.terminology.prompt_version,
            source_lang=self.settings.langs.source_name,
            domain_note=self.settings.translation.domain_note,
            segment=segment.text,
        )
        tokens_in = tokens_out = 0
        cached = True
        failures: list[str] = []
        candidates: list[TerminologyCandidate] | None = None

        for attempt in range(1 + self.settings.terminology.retry_on_parse_failure):
            body = prompt
            if attempt:
                body += (
                    f"\nValidation retry {attempt}: return valid JSON and copy every surface "
                    "exactly from the source segment."
                )
            self._check_capacity(body)
            response = self.llm.complete(
                [Message("user", body)],
                purpose="terminology_extraction",
                schema=TERMINOLOGY_SCHEMA,
                max_output_tokens=self.settings.terminology.max_output_tokens,
                temperature=self.settings.llm.temperature,
                thinking_budget=self.settings.terminology.thinking_budget,
            )
            tokens_in += response.tokens_in
            tokens_out += response.tokens_out
            cached = cached and response.cached
            try:
                parsed = CandidateResponse.model_validate(parse_json_response(response.text))
                candidates = self._validate_candidates(parsed.terms, segment.text)
            except (ParseError, ValidationError, ValueError) as exc:
                failures.append(str(exc))
                self.llm.discard_invalid(
                    [Message("user", body)],
                    schema=TERMINOLOGY_SCHEMA,
                    max_output_tokens=self.settings.terminology.max_output_tokens,
                    temperature=self.settings.llm.temperature,
                    thinking_budget=self.settings.terminology.thinking_budget,
                )
                continue
            final_prompt = body
            break
        else:
            raise TerminologyExtractionError(
                f"Terminology extraction failed for {segment.seg_id} after "
                f"{len(failures)} attempts. Last validation error: {failures[-1]}"
            )

        assert candidates is not None
        return candidates, final_prompt, tokens_in, tokens_out, cached, failures

    def _check_capacity(self, prompt: str) -> None:
        estimate = estimate_tokens(prompt, self.settings.translation.chars_per_token_estimate)
        reserved = self.settings.terminology.max_output_tokens
        limit = self.settings.terminology.request_token_limit
        if estimate + reserved > limit:
            raise TerminologyExtractionError(
                f"Estimated terminology input {estimate} + reserved output {reserved} exceeds "
                f"terminology.request_token_limit={limit}. No segment text was truncated."
            )

    @staticmethod
    def _validate_candidates(
        candidates: list[TerminologyCandidate], source_text: str
    ) -> list[TerminologyCandidate]:
        unique: list[TerminologyCandidate] = []
        seen: dict[str, str] = {}
        for candidate in candidates:
            if candidate.surface not in source_text:
                raise ValueError(
                    f"candidate surface {candidate.surface!r} is not an exact source substring"
                )
            previous = seen.get(candidate.surface)
            if previous is not None and previous != candidate.lookup_form:
                raise ValueError(
                    f"candidate surface {candidate.surface!r} has conflicting lookup forms"
                )
            if previous is None:
                seen[candidate.surface] = candidate.lookup_form
                unique.append(candidate)
        return unique

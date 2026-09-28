"""GRAFT local memory extraction and direct-predecessor memory union.

Schema and exact-key, earliest-first conflict resolution are explicit local
choices implementing the paper's five components and earlier-memory priority.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from ..config import Settings
from ..io.text import estimate_tokens
from ..llm.base import LLMError, Message, ParseError, parse_json_response
from ..llm.client import CachedLLM
from ..prompts.registry import render
from ..schemas import Segment, TranslationRecord
from .context import ContextBlock

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class MemoryPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Text
    target: Text


class DiscourseMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    noun_pronoun: list[MemoryPair]
    entities: list[MemoryPair]
    phrases: list[MemoryPair]
    connectives: list[MemoryPair]
    summary: Text


class MemoryRecord(BaseModel):
    seg_id: str
    memory: DiscourseMemory
    model: str
    prompt_version: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    parse_failures: list[str] = Field(default_factory=list)
    prompt: str = ""


# Inline schema avoids provider-specific support for Pydantic $defs/$ref.
MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            name: {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                    "required": ["source", "target"],
                },
            }
            for name in ("noun_pronoun", "entities", "phrases", "connectives")
        },
        "summary": {"type": "string"},
    },
    "required": ["noun_pronoun", "entities", "phrases", "connectives", "summary"],
}


class MemoryExtractionError(LLMError):
    pass


class RequestCapacityError(LLMError):
    pass


def check_request_capacity(prompt: str, output_tokens: int, settings: Settings) -> int:
    """Operational character estimate; no memory is silently discarded.

    This is not a provider tokenizer and excludes API framing/schema overhead.
    Configure headroom; provider context-limit errors remain visible failures.
    """
    estimate = estimate_tokens(prompt, settings.translation.chars_per_token_estimate)
    if estimate + output_tokens > settings.memory.request_token_limit:
        raise RequestCapacityError(
            f"Estimated input {estimate} + reserved output {output_tokens} exceeds "
            f"memory.request_token_limit={settings.memory.request_token_limit}. "
            "No context was truncated. Verify the model capacity and use a fresh run ID "
            "if changing the limit."
        )
    return estimate


class MemoryAgent:
    def __init__(self, settings: Settings, llm: CachedLLM):
        self.settings = settings
        self.llm = llm

    def extract(self, segment: Segment, translation: TranslationRecord) -> MemoryRecord:
        prompt = render(
            self.settings.memory.prompt_version,
            source_lang=self.settings.langs.source_name,
            target_lang=self.settings.langs.target_name,
            source=segment.text,
            translation=translation.target_text,
        )
        tokens_in = tokens_out = 0
        cached = True
        failures: list[str] = []
        for attempt in range(1 + self.settings.memory.retry_on_parse_failure):
            body = prompt
            if attempt:
                body += f"\nValidation retry {attempt}: return all five fields as valid JSON."
            check_request_capacity(body, self.settings.memory.max_output_tokens, self.settings)
            response = self.llm.complete(
                [Message("user", body)],
                purpose="memory_extraction",
                schema=MEMORY_SCHEMA,
                max_output_tokens=self.settings.memory.max_output_tokens,
                temperature=self.settings.llm.temperature,
                thinking_budget=self.settings.memory.thinking_budget,
            )
            tokens_in += response.tokens_in
            tokens_out += response.tokens_out
            cached = cached and response.cached
            try:
                memory = DiscourseMemory.model_validate(parse_json_response(response.text))
            except (ParseError, ValidationError) as exc:
                failures.append(str(exc))
                self.llm.discard_invalid(
                    [Message("user", body)],
                    schema=MEMORY_SCHEMA,
                    max_output_tokens=self.settings.memory.max_output_tokens,
                    temperature=self.settings.llm.temperature,
                    thinking_budget=self.settings.memory.thinking_budget,
                )
                continue
            return MemoryRecord(
                seg_id=segment.seg_id,
                memory=memory,
                model=self.llm.model,
                prompt_version=self.settings.memory.prompt_version,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cached=cached,
                parse_failures=failures,
                prompt=body,
            )
        raise MemoryExtractionError(
            f"Memory extraction failed for {segment.seg_id} after {len(failures)} attempts. "
            "The saved translation can be resumed; no empty memory was substituted. "
            f"Last validation error: {failures[-1]}"
        )


def merge_memories(
    parent_ids: list[str],
    segments: dict[str, Segment],
    memories: dict[str, MemoryRecord],
) -> tuple[ContextBlock, dict[str, Any]]:
    """Exact source-key union per component; earliest parent's value wins.

    Summaries are retained individually. Each winning entry identifies its
    provider; conflicting later values are saved for audit, outside the prompt.
    """
    parents = sorted(set(parent_ids), key=lambda sid: segments[sid].order)
    missing = set(parents) - memories.keys()
    if missing:
        raise MemoryExtractionError(f"Missing parent memories: {sorted(missing)}")
    merged: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    for component in ("noun_pronoun", "entities", "phrases", "connectives"):
        entries: dict[str, dict[str, str]] = {}
        for sid in parents:
            for pair in getattr(memories[sid].memory, component):
                value = {"source": pair.source, "target": pair.target, "seg_id": sid}
                if pair.source not in entries:
                    entries[pair.source] = value
                elif entries[pair.source]["target"] != pair.target:
                    conflicts.append(
                        {"component": component, "kept": entries[pair.source], "discarded": value}
                    )
        merged[component] = list(entries.values())
    merged["summaries"] = [
        {"seg_id": sid, "summary": memories[sid].memory.summary} for sid in parents
    ]
    text = json.dumps(merged, ensure_ascii=False, sort_keys=True) if parents else ""
    audit = {
        "parent_ids": parents,
        "merge_policy": "exact_key_earliest_first",
        "memory": merged,
        "conflicts": conflicts,
        "prompt_context": text,
    }
    return ContextBlock(
        text=text,
        seg_ids=parents,
        selected_seg_ids=parents,
        empty_reason=None if parents else "no_direct_parents",
    ), audit

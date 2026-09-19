"""The translation call.

One prompt template serves every strategy and both directions. The model is
asked for JSON with a single key so that a chatty preamble cannot end up in
the output, with one retry on a parse failure and a last-resort fallback that
records what happened rather than pretending it did not.
"""

from __future__ import annotations

import time

from ..config import Settings
from ..llm.base import Message, ParseError, parse_json_response
from ..llm.client import CachedLLM
from ..prompts.registry import render
from ..schemas import Segment, TranslationRecord
from . import protect
from .context import ContextBlock
from .linebreaks import LINEBREAK_TOKEN, mask_linebreaks, unmask_linebreaks

#: Structured-output schema. Providers that support it are constrained; the
#: others still see the instruction in the prompt.
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
}


class Translator:
    """Translates one segment at a time under a given context."""

    def __init__(self, settings: Settings, llm: CachedLLM) -> None:
        self.settings = settings
        self.llm = llm

    def build_prompt(self, source_text: str, context: ContextBlock) -> str:
        return render(
            self.settings.translation.prompt_version,
            source_lang=self.settings.langs.source_name,
            target_lang=self.settings.langs.target_name,
            domain_note=self.settings.translation.domain_note,
            context_block=context.text,
            segment=source_text,
            linebreak_token=LINEBREAK_TOKEN,
        )

    def translate_segment(
        self,
        segment: Segment,
        context: ContextBlock,
        strategy_name: str,
    ) -> TranslationRecord:
        # Newlines inside the segment (paragraph breaks, letter-header lines,
        # list rows) survive as literal characters up to this point, but a
        # model asked to translate and return JSON has no reason to keep them
        # in place. Masking them as an opaque token round-trips them intact
        # instead of relying on the model's judgement.
        prompt = self.build_prompt(mask_linebreaks(segment.text), context)
        started = time.perf_counter()
        text, tokens_in, tokens_out, cached, parse_flags = self._call_with_retry(prompt)
        elapsed = time.perf_counter() - started
        text = unmask_linebreaks(text)

        flags = list(parse_flags)
        report = protect.check(segment.text, text, self.settings.translation.protect)
        flags.extend(report.flags)

        return TranslationRecord(
            seg_id=segment.seg_id,
            source_text=segment.text,
            target_text=text,
            context_strategy=strategy_name,
            context_seg_ids=list(context.seg_ids),
            model=self.llm.model,
            prompt_version=self.settings.translation.prompt_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached=cached,
            latency_s=round(elapsed, 3),
            flags=flags,
            metadata={"context_truncated": context.truncated},
        )

    def _call_with_retry(self, prompt: str) -> tuple[str, int, int, bool, list[str]]:
        """Call the model, retrying once on a parse failure.

        The retry appends a reminder rather than resending the same prompt,
        because an identical prompt would be served from the cache and produce
        the identical unparseable answer.
        """
        attempts = 1 + max(0, self.settings.translation.retry_on_parse_failure)
        flags: list[str] = []
        last_text = ""
        tokens_in = tokens_out = 0
        cached = False

        for attempt in range(attempts):
            body = prompt
            if attempt > 0:
                body = (
                    prompt
                    + '\n\nYour previous answer could not be parsed. Return only a JSON '
                    'object of the form {"translation": "..."}.'
                )
            response = self.llm.complete(
                [Message("user", body)],
                schema=TRANSLATION_SCHEMA,
                purpose="translation",
                max_output_tokens=self.settings.llm.max_output_tokens,
                temperature=self.settings.llm.temperature,
            )
            tokens_in += response.tokens_in
            tokens_out += response.tokens_out
            cached = response.cached if attempt == 0 else cached and response.cached
            last_text = response.text
            try:
                value = parse_json_response(response.text, key="translation")
                if isinstance(value, str) and value.strip():
                    return value.strip(), tokens_in, tokens_out, cached, flags
                raise ParseError("translation value is empty")
            except ParseError as exc:
                flags.append(f"parse_failure_attempt_{attempt + 1}: {exc}")

        # Every attempt failed to produce JSON. Fall back to the raw text so
        # the pipeline continues and the failure is visible in the record,
        # rather than losing the document to an exception.
        flags.append("fell_back_to_raw_text")
        return last_text.strip(), tokens_in, tokens_out, cached, flags

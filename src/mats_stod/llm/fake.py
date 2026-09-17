"""Scripted LLM for offline tests.

Every test runs against this. It never opens a socket, it records what it was
asked, and it fails loudly on an unscripted prompt, so a test cannot pass by
accident on a response nobody wrote.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .base import LLMClient, LLMError, LLMResponse, Message

Responder = Callable[[list[Message], dict[str, Any]], str]


class FakeLLM(LLMClient):
    """Returns scripted responses, in order or by pattern."""

    provider = "fake"

    def __init__(
        self,
        responses: list[str] | None = None,
        rules: list[tuple[str, str]] | None = None,
        responder: Responder | None = None,
        default: str | None = None,
        model: str = "fake",
        tokens_in: int = 10,
        tokens_out: int = 5,
    ) -> None:
        """
        `responses` is a queue consumed in order; `rules` maps a regex over the
        rendered prompt to a canned answer; `responder` is an arbitrary
        function for tests that need to look at the prompt.
        """
        self.model = model
        self._queue = list(responses or [])
        self._rules = [(re.compile(p, re.DOTALL), r) for p, r in (rules or [])]
        self._responder = responder
        self._default = default
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        #: Every call, for assertions on call counts and prompt content.
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        **params: Any,
    ) -> LLMResponse:
        prompt = "\n\n".join(m.content for m in messages)
        self.calls.append(
            {"messages": [m.as_dict() for m in messages], "schema": schema, "params": params}
        )

        text = self._resolve(messages, prompt, params)
        return LLMResponse(
            text=text,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            cached=False,
            model=self.model,
        )

    def _resolve(self, messages: list[Message], prompt: str, params: dict[str, Any]) -> str:
        if self._responder is not None:
            return self._responder(messages, params)
        for pattern, reply in self._rules:
            if pattern.search(prompt):
                return reply
        if self._queue:
            return self._queue.pop(0)
        if self._default is not None:
            return self._default
        raise LLMError(
            "FakeLLM has no scripted response for this prompt. "
            f"First 300 chars: {prompt[:300]!r}"
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def prompts(self) -> list[str]:
        return ["\n\n".join(m["content"] for m in c["messages"]) for c in self.calls]


class EchoLLM(FakeLLM):
    """An offline stand-in that produces structurally valid, useless output.

    It echoes the source back as the "translation" and answers every yes/no
    question with "no". That makes `--provider fake` usable for smoke-testing
    a command without a credential, while guaranteeing the numbers it produces
    are obviously not translations: chrF++ against a real reference collapses,
    so such a run can never be mistaken for a result.
    """

    provider = "fake"

    def __init__(self, model: str = "fake") -> None:
        super().__init__(responder=_echo, model=model)


def _echo(messages: list[Message], params: dict[str, Any]) -> str:
    import json

    prompt = "\n\n".join(m.content for m in messages)
    if "Decision:" in prompt:
        return "no"
    marker = "Source text"
    if marker in prompt:
        tail = prompt.split(marker, 1)[1]
        body = tail.split(":", 1)[1] if ":" in tail else tail
        body = body.split("Return a JSON", 1)[0].strip()
        return json.dumps({"translation": body}, ensure_ascii=False)
    return json.dumps({"translation": ""}, ensure_ascii=False)

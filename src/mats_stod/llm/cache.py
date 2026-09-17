"""Disk cache for LLM calls.

The token budget is a free trial, and Sinhala and Tamil tokenise expensively,
so every call is cached on disk keyed by everything that could change the
answer. Re-running an experiment after a code change that does not touch
prompts costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .base import LLMResponse, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key         TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    response    TEXT NOT NULL,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def cache_key(
    provider: str,
    model: str,
    messages: list[Message],
    schema: dict[str, Any] | None,
    params: dict[str, Any],
) -> str:
    """Stable hash over everything that determines the response.

    Parameters are sorted so that keyword order cannot produce a cache miss on
    an otherwise identical call.
    """
    payload = {
        "provider": provider,
        "model": model,
        "messages": [m.as_dict() for m in messages],
        "schema": schema,
        "params": {k: params[k] for k in sorted(params)},
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    """SQLite-backed response cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, tokens_in, tokens_out, model FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return LLMResponse(
            text=row[0],
            tokens_in=row[1],
            tokens_out=row[2],
            cached=True,
            model=row[3],
        )

    def put(self, key: str, provider: str, model: str, response: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(key, provider, model, response, tokens_in, tokens_out) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, provider, model, response.text, response.tokens_in, response.tokens_out),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

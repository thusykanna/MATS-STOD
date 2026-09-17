"""Run directories.

Every command writes into `runs/<run_id>/`. A run holds the exact config that
produced it, every intermediate artifact as JSON, a JSONL event log and the
cost ledger, so a result can be re-examined months later without re-running
anything.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config import REPO_ROOT, Settings


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_run_id(prefix: str) -> str:
    """Timestamped, sortable run id."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{prefix}"


class RunDir:
    """Handle on one run directory."""

    def __init__(self, path: Path, settings: Settings, dry_run: bool = False) -> None:
        self.path = path
        self.settings = settings
        self.dry_run = dry_run
        self.path.mkdir(parents=True, exist_ok=True)
        self._log_path = self.path / "log.jsonl"

    @classmethod
    def create(
        cls,
        settings: Settings,
        prefix: str,
        run_id: str | None = None,
        dry_run: bool = False,
    ) -> RunDir:
        rid = run_id or new_run_id(prefix)
        run = cls(Path(settings.paths.runs) / rid, settings, dry_run=dry_run)
        run.write_text("config_used.yaml", settings.to_yaml())
        run.write_json(
            "run_meta.json",
            {
                "run_id": rid,
                "created_utc": datetime.now(UTC).isoformat(),
                "git_commit": _git_commit(),
                "direction": settings.langs.direction,
                "dry_run": dry_run,
                "env": {
                    "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    "GOOGLE_CLOUD_LOCATION": os.environ.get("GOOGLE_CLOUD_LOCATION"),
                },
            },
        )
        return run

    @property
    def run_id(self) -> str:
        return self.path.name

    # -- writers ----------------------------------------------------------

    def write_text(self, name: str, text: str) -> Path:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def write_json(self, name: str, data: Any) -> Path:
        payload = data
        if isinstance(data, BaseModel):
            payload = data.model_dump(mode="json")
        elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
            payload = [d.model_dump(mode="json") for d in data]
        return self.write_text(
            name, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
        )

    def read_json(self, name: str) -> Any:
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def log(self, event: str, **fields: Any) -> None:
        """Append one structured event.

        Timestamps stay out of artifact JSON so that two runs on identical
        input produce byte-identical artifacts; they live here instead.
        """
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def subdir(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

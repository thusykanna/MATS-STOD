"""Fixed dev/test split.

The split is written once to a file and read from there forever. It is never
reshuffled, because a split that moves makes every earlier number
incomparable, and the panel will ask whether a gain came from the method or
from a luckier test set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class SplitError(RuntimeError):
    pass


def create_split(
    doc_ids: list[str],
    path: str | Path,
    dev_fraction: float = 0.5,
    seed: int = 12345,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Create the split file. Refuses to overwrite silently."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise SplitError(
            f"{p} already exists. The split is fixed by design; pass overwrite=True "
            "only if you accept that all previous results become incomparable."
        )
    ordered = sorted(set(doc_ids))
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_dev = max(1, round(len(shuffled) * dev_fraction)) if shuffled else 0
    split = {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "dev": sorted(shuffled[:n_dev]),
        "test": sorted(shuffled[n_dev:]),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split


def load_split(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SplitError(f"split file not found: {p}. Create it with `mats-stod make-split`.")
    return json.loads(p.read_text(encoding="utf-8"))


def select(
    doc_ids: list[str], path: str | Path, portion: str = "dev", strict: bool = True
) -> list[str]:
    """Return the requested portion, in the order given by the split file."""
    if portion == "all":
        return sorted(doc_ids)
    split = load_split(path)
    if portion not in {"dev", "test"}:
        raise SplitError(f"unknown portion: {portion!r}")
    wanted = split[portion]
    available = set(doc_ids)
    missing = [d for d in wanted if d not in available]
    if missing and strict:
        raise SplitError(f"documents in the {portion} split are missing from the corpus: {missing}")
    return [d for d in wanted if d in available]

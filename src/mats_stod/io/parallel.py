"""Load aligned pairs and source-only translation inputs.

The corpus is a directory per document. Strict evaluation loading requires one
Sinhala file and one Tamil file. Translation discovery requires only the file
for the selected source language and attaches the other file as an optional
reference when it exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .text import read_text

#: Accepted file naming conventions inside a pair directory, tried in order.
#: `source.txt`/`reference.txt` is the layout named in the build prompt;
#: `<name>.si`/`<name>.ta` follows GRAFT's corpus layout.
_LANG_SUFFIXES = ("si", "ta", "en")


@dataclass
class SegmentAlignment:
    """One aligned source/target segment pair, when alignment is available."""

    index: int
    source_text: str
    target_text: str


@dataclass
class DocPair:
    """One document in one translation direction.

    ``reference_text`` is absent for source-only translation inputs.  Strict
    corpus and evaluation loaders still return pairs with references.
    """

    doc_id: str
    source_lang: str
    target_lang: str
    source_text: str
    reference_text: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    alignment: list[SegmentAlignment] | None = None
    source_path: Path | None = None
    reference_path: Path | None = None

    @property
    def direction(self) -> str:
        return f"{self.source_lang}-{self.target_lang}"

    @property
    def key(self) -> str:
        """Unique id of this document in this direction."""
        return f"{self.doc_id}::{self.direction}"

    @property
    def has_reference(self) -> bool:
        """Whether this document can be scored against a target reference."""
        return self.reference_text is not None


@dataclass(frozen=True)
class SkippedDocument:
    """A directory that could not supply a source for the selected direction."""

    doc_id: str
    reason: str


@dataclass
class TranslationDiscovery:
    """Translation inputs plus the diagnostics needed for CLI feedback."""

    root: Path
    discovered_count: int
    valid_count: int
    pairs: list[DocPair]
    skipped: list[SkippedDocument] = field(default_factory=list)


def _read_meta(pair_dir: Path) -> dict[str, Any]:
    meta_path = pair_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _read_alignment(
    pair_dir: Path, source_lang: str, target_lang: str
) -> list[SegmentAlignment] | None:
    """Read alignment.jsonl if present.

    Each line is an object with one key per language code, so the same file
    serves both directions.
    """
    path = pair_dir / "alignment.jsonl"
    if not path.exists():
        return None
    out: list[SegmentAlignment] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if source_lang not in row or target_lang not in row:
            # Alignment exists but not for this direction; treat as absent
            # rather than guessing which column is which.
            return None
        out.append(
            SegmentAlignment(index=i, source_text=row[source_lang], target_text=row[target_lang])
        )
    return out or None


def _find_lang_files(pair_dir: Path) -> dict[str, Path]:
    """Map language code -> file inside one pair directory."""
    found: dict[str, Path] = {}
    for path in sorted(pair_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in {"meta.json", "alignment.jsonl"}:
            continue
        stem_suffix = path.suffix.lstrip(".").lower()
        if stem_suffix in _LANG_SUFFIXES:  # doc.si / doc.ta
            found.setdefault(stem_suffix, path)
            continue
        if path.suffix.lower() == ".txt":
            base = path.stem.lower()
            for lang in _LANG_SUFFIXES:
                if base == lang or base.endswith(f"_{lang}") or base.endswith(f".{lang}"):
                    found.setdefault(lang, path)
                    break
    return found


def load_pair_dir(pair_dir: Path, directions: list[tuple[str, str]]) -> list[DocPair]:
    """Load one pair directory into one DocPair per requested direction."""
    doc_id = pair_dir.name
    meta = _read_meta(pair_dir)
    files = _find_lang_files(pair_dir)

    # Fall back to the explicit source/reference layout, which names roles
    # rather than languages; meta.json must then say which language is which.
    if not files and (pair_dir / "source.txt").exists():
        src_lang = meta.get("source_lang")
        tgt_lang = meta.get("target_lang")
        if not src_lang or not tgt_lang:
            raise ValueError(
                f"{pair_dir}: source.txt/reference.txt layout requires "
                "meta.json with source_lang and target_lang"
            )
        files = {src_lang: pair_dir / "source.txt", tgt_lang: pair_dir / "reference.txt"}

    out: list[DocPair] = []
    for src_lang, tgt_lang in directions:
        if src_lang not in files or tgt_lang not in files:
            continue
        out.append(
            DocPair(
                doc_id=doc_id,
                source_lang=src_lang,
                target_lang=tgt_lang,
                source_text=read_text(files[src_lang]),
                reference_text=read_text(files[tgt_lang]),
                metadata=dict(meta),
                alignment=_read_alignment(pair_dir, src_lang, tgt_lang),
                source_path=files[src_lang],
                reference_path=files[tgt_lang],
            )
        )
    return out


def _load_translation_dir(
    pair_dir: Path, source_lang: str, target_lang: str
) -> tuple[DocPair | None, str | None]:
    """Load one directory while allowing its target reference to be absent."""
    try:
        meta = _read_meta(pair_dir)
        files = _find_lang_files(pair_dir)

        # ``source.txt`` names a role rather than a language, so metadata is
        # necessary even when no reference is available.
        if not files and (pair_dir / "source.txt").exists():
            meta_source = meta.get("source_lang")
            meta_target = meta.get("target_lang")
            if not meta_source or not meta_target:
                raise ValueError(
                    "source.txt layout requires meta.json with source_lang and target_lang"
                )
            files[meta_source] = pair_dir / "source.txt"
            if (pair_dir / "reference.txt").exists():
                files[meta_target] = pair_dir / "reference.txt"

        source_path = files.get(source_lang)
        if source_path is None:
            if files:
                available = ", ".join(sorted(files))
                return None, (
                    f"no {source_lang} source file for {source_lang}-{target_lang}; "
                    f"found language file(s): {available}"
                )
            return None, "no recognized language source file"

        reference_path = files.get(target_lang)
        pair = DocPair(
            doc_id=pair_dir.name,
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=read_text(source_path),
            reference_text=read_text(reference_path) if reference_path is not None else None,
            metadata=dict(meta),
            alignment=(
                _read_alignment(pair_dir, source_lang, target_lang)
                if reference_path is not None
                else None
            ),
            source_path=source_path,
            reference_path=reference_path,
        )
        return pair, None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)


def discover_translation_inputs(
    root: str | Path,
    source_lang: str,
    target_lang: str,
    doc_ids: list[str] | None = None,
    max_docs: int | None = None,
) -> TranslationDiscovery:
    """Discover source documents and optional references for translation.

    Every candidate directory is inspected before ``max_docs`` is applied, so
    the limit always counts usable source documents rather than raw folders.
    """
    if max_docs is not None and max_docs < 1:
        raise ValueError("max_docs must be at least 1")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"parallel data directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"parallel data path is not a directory: {root_path}")

    wanted = set(doc_ids) if doc_ids is not None else None
    directories = [
        path
        for path in sorted(root_path.iterdir())
        if path.is_dir() and (wanted is None or path.name in wanted)
    ]
    valid: list[DocPair] = []
    skipped: list[SkippedDocument] = []
    for pair_dir in directories:
        pair, reason = _load_translation_dir(pair_dir, source_lang, target_lang)
        if pair is None:
            skipped.append(SkippedDocument(pair_dir.name, reason or "unusable input"))
        else:
            valid.append(pair)

    selected = valid[:max_docs] if max_docs is not None else valid
    return TranslationDiscovery(
        root=root_path,
        discovered_count=len(directories),
        valid_count=len(valid),
        pairs=selected,
        skipped=skipped,
    )


def load_parallel(
    root: str | Path,
    source_lang: str,
    target_lang: str,
    doc_ids: list[str] | None = None,
    max_docs: int | None = None,
) -> list[DocPair]:
    """Load every pair directory under `root` for one direction.

    Ordering is by directory name so that `--max-docs 3` selects the same three
    documents on every run.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"parallel data directory not found: {root_path}")

    pairs: list[DocPair] = []
    for pair_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if doc_ids is not None and pair_dir.name not in doc_ids:
            continue
        pairs.extend(load_pair_dir(pair_dir, [(source_lang, target_lang)]))

    if max_docs is not None:
        pairs = pairs[:max_docs]
    return pairs


def load_text_dir(root: str | Path, suffix: str = ".txt") -> dict[str, str]:
    """Load a flat directory of translations keyed by file stem.

    Used by `mats-stod eval --hyp <dir> --ref <dir>`, where the two directories
    are matched on file stem.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"directory not found: {root_path}")
    return {
        p.stem: read_text(p)
        for p in sorted(root_path.iterdir())
        if p.is_file() and p.suffix == suffix
    }

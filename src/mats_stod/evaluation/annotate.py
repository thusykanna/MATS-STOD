"""Gold-annotation helper.

Hand-annotating boundaries and edges in raw text is slow and error-prone, and
character offsets cannot be typed by a human at all. The template writes one
numbered sentence per line so a boundary is marked by editing a single
character, and the importer converts those marks back into offsets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import Settings
from ..parsing.plaintext import PlainTextParser
from ..schemas import Document
from ..segmentation.sentences import SentenceSplitter

_HEADER = """\
# Gold annotation: {doc_id} ({source_lang} source)
#
# Sentence units are listed below, one per line, as:
#     <unit_id>\t<B|->\t<char_start>:<char_end>\t<text>
#
# Boundaries: put B in the second column of every unit that STARTS a new
# segment. The first unit is always a start and is marked B already.
#
# Edges: add one line per dependency under the EDGES heading, as:
#     <src_unit_id>\t<dst_unit_id>\t<type>\t<evidence text>
# where src is the unit depended on and dst is the unit that depends on it,
# src must come before dst, and evidence must be text copied from the SOURCE
# segment, not paraphrased.
#
# Edge types: {edge_types}
#
# Lines starting with # are ignored. Do not edit the offsets or the text.

UNITS
"""

_EDGES_MARKER = "EDGES"


def build_template(document: Document, settings: Settings) -> str:
    """Render a human-editable annotation file for one document."""
    splitter = SentenceSplitter(settings.segmentation.sentences)
    lines = [
        _HEADER.format(
            doc_id=document.doc_id,
            source_lang=document.source_lang,
            edge_types=", ".join(settings.graph.edge_types),
        )
    ]
    index = 0
    for block in document.blocks or []:
        for span in splitter.split(block.text_of(document.raw_text), offset=block.char_start):
            text = span.text_of(document.raw_text).replace("\n", " ⏎ ")
            mark = "B" if index == 0 else "-"
            lines.append(f"u{index:04d}\t{mark}\t{span.char_start}:{span.char_end}\t{text}")
            index += 1
    lines.append("")
    lines.append(_EDGES_MARKER)
    lines.append("# src\tdst\ttype\tevidence")
    lines.append("")
    return "\n".join(lines)


def write_template(document: Document, settings: Settings, out_dir: str | Path) -> Path:
    """Write the template under a language-tagged name.

    The same document is a source in one direction and a reference in the
    other, and its segmentation differs between the two. Tagging the file with
    the source language is what stops Sinhala gold being scored against a
    Tamil segmentation.
    """
    path = Path(out_dir) / f"{gold_stem(document.doc_id, document.source_lang)}.annot.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_template(document, settings), encoding="utf-8")
    return path


def gold_stem(doc_id: str, source_lang: str) -> str:
    """The name a gold file has for one document in one source language."""
    return f"{doc_id}.{source_lang}"


_UNIT_LINE = re.compile(r"^(u\d+)\t([B\-])\t(\d+):(\d+)\t(.*)$")


def parse_annotation(text: str) -> dict[str, Any]:
    """Parse a filled-in annotation file into gold boundaries and edges."""
    units: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    section = "units"

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.strip() == "UNITS":
            section = "units"
            continue
        if line.strip() == _EDGES_MARKER:
            section = "edges"
            continue

        if section == "units":
            m = _UNIT_LINE.match(line)
            if not m:
                raise ValueError(f"malformed unit line: {line[:120]!r}")
            unit_id, mark, start, end, body = m.groups()
            units.append(
                {
                    "unit_id": unit_id,
                    "is_boundary": mark.upper() == "B",
                    "char_start": int(start),
                    "char_end": int(end),
                    "text": body,
                }
            )
        else:
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"malformed edge line: {line[:120]!r}")
            edges.append(
                {
                    "src": parts[0].strip(),
                    "dst": parts[1].strip(),
                    "type": parts[2].strip(),
                    "evidence": parts[3].strip() if len(parts) > 3 else "",
                }
            )

    if not units:
        raise ValueError("annotation file contains no units")

    # The first unit starts the first segment; it is not an internal boundary.
    boundaries = [u["char_start"] for u in units[1:] if u["is_boundary"]]
    return {"units": units, "boundaries": sorted(boundaries), "edges": edges}


def import_annotation(
    annot_path: str | Path,
    gold_segmentation_dir: str | Path,
    gold_edges_dir: str | Path,
) -> tuple[Path, Path]:
    """Convert a filled-in template into the two gold JSON files."""
    path = Path(annot_path)
    stem = path.name.replace(".annot.tsv", "")
    doc_id, _, lang = stem.partition(".")
    if not lang:
        raise ValueError(
            f"{path.name}: annotation files must be named <doc_id>.<lang>.annot.tsv so that "
            "gold is never scored against the wrong source language"
        )
    parsed = parse_annotation(path.read_text(encoding="utf-8"))

    seg_path = Path(gold_segmentation_dir) / f"{stem}.json"
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.write_text(
        json.dumps(
            {"doc_id": doc_id, "source_lang": lang, "boundaries": parsed["boundaries"]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Edges reference unit ids; the segment they fall into depends on the
    # boundaries in the same file, so resolve them here rather than later.
    unit_to_segment = _unit_segment_map(parsed["units"])
    resolved = []
    for e in parsed["edges"]:
        if e["src"] not in unit_to_segment or e["dst"] not in unit_to_segment:
            raise ValueError(f"edge references an unknown unit: {e}")
        resolved.append(
            {
                "src_segment_index": unit_to_segment[e["src"]],
                "dst_segment_index": unit_to_segment[e["dst"]],
                "src_unit": e["src"],
                "dst_unit": e["dst"],
                "type": e["type"],
                "evidence": e["evidence"],
            }
        )

    edge_path = Path(gold_edges_dir) / f"{stem}.json"
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.write_text(
        json.dumps(
            {"doc_id": doc_id, "source_lang": lang, "edges": resolved}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return seg_path, edge_path


def _unit_segment_map(units: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    seg_index = -1
    for i, u in enumerate(units):
        if i == 0 or u["is_boundary"]:
            seg_index += 1
        mapping[u["unit_id"]] = seg_index
    return mapping


def parse_document_for_annotation(text: str, doc_id: str, source_lang: str) -> Document:
    return PlainTextParser().parse(text, doc_id, source_lang)

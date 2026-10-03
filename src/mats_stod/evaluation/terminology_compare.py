"""Compare completed terminology experiment runs without rerunning translation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class TerminologyComparisonError(ValueError):
    pass


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise TerminologyComparisonError(f"required run artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _graph_hash(path: Path) -> str:
    payload = json.dumps(_load_json(path), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def compare_terminology_runs(run_paths: list[str | Path]) -> dict[str, Any]:
    if len(run_paths) < 2:
        raise TerminologyComparisonError("provide at least two completed run directories")
    runs: list[dict[str, Any]] = []
    for raw_path in run_paths:
        path = Path(raw_path)
        config_path = path / "config_used.yaml"
        if not config_path.exists():
            raise TerminologyComparisonError(f"required run artifact is missing: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        results = _load_json(path / "results.json")
        glossary = _load_json(path / "glossary_snapshot.json")
        documents_root = path / "documents"
        documents = sorted(p.name for p in documents_root.iterdir() if p.is_dir())
        graphs = {
            doc_id: _graph_hash(documents_root / doc_id / "graph.json") for doc_id in documents
        }
        runs.append({"path": str(path), "config": config, "results": results,
                     "glossary": glossary, "documents": documents, "graphs": graphs})

    first = runs[0]
    compatibility = {
        "direction": first["results"]["direction"],
        "model": first["config"]["llm"]["model"],
        "glossary_hash": first["glossary"]["sha256"],
        "documents": first["documents"],
        "segmentation_prompt": first["config"]["segmentation"]["prompt_version"],
        "graph_prompt": first["config"]["graph"]["prompt_version"],
        "translation_prompt": first["config"]["translation"]["prompt_version"],
        "terminology_prompt": first["config"]["terminology"]["prompt_version"],
    }
    for run in runs[1:]:
        checks = {
            "direction": run["results"]["direction"],
            "model": run["config"]["llm"]["model"],
            "glossary_hash": run["glossary"]["sha256"],
            "documents": run["documents"],
            "segmentation_prompt": run["config"]["segmentation"]["prompt_version"],
            "graph_prompt": run["config"]["graph"]["prompt_version"],
            "translation_prompt": run["config"]["translation"]["prompt_version"],
            "terminology_prompt": run["config"]["terminology"]["prompt_version"],
        }
        mismatches = [key for key, value in checks.items() if value != compatibility[key]]
        if run["graphs"] != first["graphs"]:
            mismatches.append("graphs")
        if mismatches:
            raise TerminologyComparisonError(
                f"incompatible run {run['path']}: {', '.join(mismatches)} differ"
            )

    rows = []
    for run in runs:
        results = run["results"]
        corpus = results.get("corpus") or {}
        term = results.get("terminology_evaluation") or {}
        calls = ((results.get("calls") or {}).get("by_purpose") or {}).get(
            "terminology_extraction", {}
        )
        operations = results.get("terminology_operational") or {}
        rows.append({
            "run": run["path"],
            "method": run["config"]["terminology"]["method"],
            "chrf": corpus.get("chrf"),
            "bleu": corpus.get("bleu"),
            "comet": (corpus.get("extra") or {}).get("comet"),
            "identification_f1": (term.get("identification") or {}).get("f1"),
            "resolution_precision": (term.get("resolution") or {}).get("precision"),
            "resolution_recall": (term.get("resolution") or {}).get("recall"),
            "false_mappings": (term.get("resolution") or {}).get("false_mappings"),
            "recovery_recall": (term.get("recovery") or {}).get("recall"),
            "target_realization": (term.get("target_realization") or {}).get("accuracy"),
            "terminology_calls": calls.get("calls", 0),
            "terminology_cache_hits": calls.get("cached", 0),
            "terminology_tokens_in": calls.get("tokens_in", 0),
            "terminology_tokens_out": calls.get("tokens_out", 0),
            "terminology_latency_s": operations.get("latency_s", 0.0),
        })
    return {"compatibility": compatibility, "conditions": rows}


def render_terminology_comparison(payload: dict[str, Any]) -> str:
    headers = [
        "method", "chrF++", "BLEU", "COMET", "term F1", "resolution P",
        "resolution R", "false maps", "recovery R", "target accuracy", "term calls",
        "cache hits", "tokens in", "tokens out",
        "term latency s",
    ]
    lines = ["# Terminology method comparison\n", "| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in payload["conditions"]:
        values = [row[key] for key in (
            "method", "chrf", "bleu", "comet", "identification_f1",
            "resolution_precision", "resolution_recall", "false_mappings",
            "recovery_recall", "target_realization", "terminology_calls",
            "terminology_cache_hits", "terminology_tokens_in", "terminology_tokens_out",
            "terminology_latency_s",
        )]
        cells = " | ".join("" if value is None else str(value) for value in values)
        lines.append(f"| {cells} |")
    return "\n".join(lines) + "\n"

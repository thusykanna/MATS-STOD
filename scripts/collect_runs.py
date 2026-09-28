"""Collect MATS-STOD run results into one Excel workbook.

Scans runs/*/results.json and config_used.yaml. Two kinds of run are recognised
by the text after the timestamp in the folder name:

    translate*  ->  sheet "Runs"   (chrF++, BLEU, calls)
    graph*      ->  sheet "Graph"  (segments, edges, depth, gold metrics if any)

The workbook is rebuilt from the run folders every time, so do not type into it.

Usage (from the project root, no permanent install needed):
    uv run --with openpyxl python scripts/collect_runs.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FONT = "Arial"

TRANSLATE_COLS = [
    "run_id", "direction", "model", "prompts", "docs",
    "chrF++", "BLEU", "COMET",
    "calls", "cache hits", "tokens in", "tokens out", "latency (s)",
    "segmentation calls", "edge calls", "translation calls",
]
GRAPH_COLS = [
    "run_id", "direction", "model", "prompts", "segmenter", "edge inferrer",
    "docs", "segments", "edges", "adjacent-only share", "non-adjacent parent share",
    "forward refs", "max depth", "calls", "segmentation calls", "edge calls",
    "other metrics", "run notes",
]

# Top-level results.json keys already shown in dedicated columns.
KNOWN_KEYS = {
    "run_name", "direction", "model", "model_configured", "corpus",
    "documents", "calls", "graph_stats", "notes", "flagged_segments",
}

PROMPT_VALUE = re.compile(r"(discourse|edge|memory|translate)[_/]v\d", re.I)


def flatten(node, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = node
    return out


def load_run(path: Path) -> dict | None:
    results_file = path / "results.json"
    if not results_file.exists():
        return None
    res = json.loads(results_file.read_text(encoding="utf-8"))
    cfg_file = path / "config_used.yaml"
    cfg = {}
    if cfg_file.exists():
        cfg = flatten(yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {})
    match = re.match(r"^\d{8}T\d{6}Z_(.*)$", path.name)
    return {
        "run_id": path.name,
        "type": match.group(1) if match else path.name,
        "res": res,
        "cfg": cfg,
    }


def purpose_calls(res: dict, purpose: str):
    calls = res.get("calls") or {}
    return ((calls.get("by_purpose") or {}).get(purpose) or {}).get("calls")


def prompts_used(cfg: dict) -> str | None:
    """Config entries that name a prompt: key contains 'prompt' or value looks like translate/v2_x."""
    parts = [
        f"{k}={v}" for k, v in sorted(cfg.items())
        if "prompt" in k.lower() or PROMPT_VALUE.search(str(v))
    ]
    return "; ".join(parts) or None


def translate_row(run: dict) -> dict:
    res = run["res"]
    corpus = res.get("corpus") or {}
    calls = res.get("calls") or {}
    doc_ids = ", ".join(d.get("doc_id", "?") for d in res.get("documents") or [])
    return {
        "run_id": run["run_id"],
        "direction": res.get("direction"),
        "model": res.get("model"),
        "prompts": prompts_used(run["cfg"]),
        "docs": f"{corpus.get('n_docs')} ({doc_ids})",
        "chrF++": corpus.get("chrf"),
        "BLEU": corpus.get("bleu"),
        "COMET": (corpus.get("extra") or {}).get("comet"),
        "calls": calls.get("calls"),
        "cache hits": calls.get("cache_hits"),
        "tokens in": calls.get("tokens_in_total"),
        "tokens out": calls.get("tokens_out_total"),
        "latency (s)": calls.get("latency_s_total"),
        "segmentation calls": purpose_calls(res, "segmentation"),
        "edge calls": purpose_calls(res, "edge_inference"),
        "translation calls": purpose_calls(res, "translation"),
    }


def graph_row(run: dict) -> dict:
    res = run["res"]
    gs = res.get("graph_stats") or {}
    calls = res.get("calls") or {}
    extra = {k: v for k, v in res.items() if k not in KNOWN_KEYS}
    return {
        "run_id": run["run_id"],
        "direction": res.get("direction"),
        "model": res.get("model"),
        "prompts": prompts_used(run["cfg"]),
        "segmenter": gs.get("segmenter"),
        "edge inferrer": gs.get("edge_inferrer"),
        "docs": gs.get("n_documents"),
        "segments": gs.get("n_segments_total"),
        "edges": gs.get("n_edges_total"),
        "adjacent-only share": gs.get("adjacent_only_share"),
        "non-adjacent parent share": gs.get("share_with_nonadjacent_parent"),
        "forward refs": gs.get("n_forward_refs_total"),
        "max depth": gs.get("depth_max"),
        "calls": calls.get("calls"),
        "segmentation calls": purpose_calls(res, "segmentation"),
        "edge calls": purpose_calls(res, "edge_inference"),
        "other metrics": json.dumps(extra, ensure_ascii=False) if extra else None,
        "run notes": " ".join(res.get("notes") or []) or None,
    }


def write_sheet(wb, title, runs, headers, make_row, number_cols, widths):
    ws = wb.create_sheet(title)
    ws.append(headers)
    for run in runs:
        row = make_row(run)
        ws.append([row.get(h) for h in headers])

    col = {name: get_column_letter(i + 1) for i, name in enumerate(headers)}
    fill = PatternFill("solid", start_color="D9D9D9")
    for c in ws[1]:
        c.font = Font(name=FONT, bold=True)
        c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for name in number_cols:
        for r in range(2, len(runs) + 2):
            ws[f"{col[name]}{r}"].number_format = "0.00"
    for name, letter in col.items():
        ws.column_dimensions[letter].width = widths.get(name, 13)
    ws.freeze_panes = "B2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs", help="runs directory")
    ap.add_argument("--out", default="experiments.xlsx", help="output workbook")
    ap.add_argument("--types", default="translate,graph",
                    help="comma-separated run kinds to include: translate, graph")
    ap.add_argument("--include-fake", action="store_true",
                    help="include runs made with the offline fake model (placeholder scores)")
    args = ap.parse_args()

    wanted = {t.strip() for t in args.types.split(",") if t.strip()}
    groups: dict[str, list] = {"translate": [], "graph": []}
    for path in sorted(Path(args.runs).iterdir()):
        if not path.is_dir():
            continue
        run = load_run(path)
        if run is None:
            continue
        kind = next((k for k in groups if run["type"].startswith(k)), None)
        if kind is None or kind not in wanted:
            continue
        if not args.include_fake and "fake" in str(run["res"].get("model", "")).lower():
            continue
        groups[kind].append(run)

    if not any(groups.values()):
        raise SystemExit(f"No matching runs found in {args.runs!r} for {sorted(wanted)}.")

    wb = Workbook()
    wb.remove(wb.active)

    if groups["translate"]:
        write_sheet(wb, "Runs", groups["translate"], TRANSLATE_COLS, translate_row,
                    ["chrF++", "BLEU", "COMET"], {"run_id": 34, "prompts": 36, "docs": 22})
    if groups["graph"]:
        write_sheet(wb, "Graph", groups["graph"], GRAPH_COLS, graph_row,
                    [], {"run_id": 34, "prompts": 36, "other metrics": 40, "run notes": 40})

    legend = wb.create_sheet("Legend")
    lines = [
        "How to read this workbook",
        "",
        "Sheet 'Runs': translate* runs, scored on translation quality (chrF++, BLEU). A translate-dag run also does its own",
        "  segmentation and edge inference, so prompt changes to those agents show up here as changes in chrF++ and call counts.",
        "Sheet 'Graph': graph* runs. These report graph structure (segments, edges, depth). Correctness metrics (precision, recall,",
        "  F1) appear in 'other metrics' only when gold annotations exist; otherwise 'run notes' says so.",
        "Everything is read from runs/*/results.json and config_used.yaml, and rebuilt on every run of the script.",
        "'prompts' lists the config entries that name a prompt (key contains 'prompt', or value like translate/v2_x).",
        "  If it is empty, the config does not record prompts.",
        "Fake-model runs are skipped unless --include-fake is passed; their scores and graph shapes are placeholders.",
        "Cost is not in results.json; see report.md in each run folder.",
        "Samples in data/samples are synthetic: do not report scores from them as results.",
    ]
    for line in lines:
        legend.append([line])
    legend["A1"].font = Font(name=FONT, bold=True)
    for r in range(2, len(lines) + 1):
        legend[f"A{r}"].font = Font(name=FONT)
    legend.column_dimensions["A"].width = 125

    wb.save(args.out)
    summary = ", ".join(f"{len(v)} {k}" for k, v in groups.items() if v)
    print(f"Wrote {summary} run(s) to {args.out}")


if __name__ == "__main__":
    main()
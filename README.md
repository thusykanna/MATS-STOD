# MATS-STOD

Multi-Agent Translation System for Sinhala–Tamil Official Documents.

A research instrument, not a product: every design choice is switchable from
config, measurable, and recorded in [DECISIONS.md](DECISIONS.md).

The segmentation and graph method reuses **GRAFT** (Dutta et al., *GRAFT: A
Graph-based Flow-aware Agentic Framework for Document-level Machine
Translation*, EMNLP 2025 Industry Track, [arXiv
2507.03311](https://arxiv.org/abs/2507.03311)).

## Status

| Stage | What it covers | State |
|---|---|---|
| M0 | Scaffold, schemas, LLM cache, cost ledger, metrics, reports | Done |
| M1 | Baseline translation: B0, B1, B2 | Done |
| M2 | Segmentation: structural and GRAFT discourse agent | Done |
| M3 | Edge inference and DAG assembly | Not started |
| M4 | DAG-context translation and the ablation | Not started |

Both translation directions are supported everywhere: `si→ta` and `ta→si`.

## Install

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --dev                 # core + test tooling
uv sync --extra gemini        # add when you want to call a real model
```

Everything below runs offline with `--provider fake`. No test touches the
network; a fixture blocks sockets for the whole suite.

## Quick start

```bash
uv run pytest                                   # 148 tests, ~1s, offline
uv run mats-stod make-split --data data/samples # throwaway split over the samples
uv run mats-stod compare --strategies B0,B1,B2 \
    --provider fake --data data/samples --portion all
```

Output lands in `runs/<run_id>/`: `report.md`, `results.json`,
`config_used.yaml`, per-document artifacts and a JSONL event log.

## Using a real model

Gemini on Vertex AI is the default backend.

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
uv run mats-stod translate --strategy B2 --max-docs 2
```

To use an AI Studio key instead, set `llm.backend: ai_studio` in the config and
export `GEMINI_API_KEY`. Both backends produce identical cache keys, so
switching never re-spends tokens.

Budget controls, available on every command:

- `--max-docs N` caps how many documents are processed.
- `--dry-run` serves cached calls and **fails** on the first uncached one,
  rather than inventing an answer.
- Every call goes through a SQLite cache, so a re-run costs nothing. The report
  prints both what the run spent and what it would cost from a cold cache.

## Commands

| Command | Purpose |
|---|---|
| `translate --strategy B0\|B1\|B2` | Translate under one context strategy |
| `compare --strategies B0,B1,B2` | Run several conditions, one comparison table |
| `segment --segmenter structural\|graft` | Segment and score against gold |
| `eval --hyp DIR --ref DIR` | Score two directories of `.txt` files |
| `annotate-template DOC` | Write a hand-editable gold annotation file |
| `annotate-import FILE` | Convert a filled-in template into gold JSON |
| `make-split` | Create the fixed dev/test split, once |

`data/splits.json` is gitignored because it is derived from the corpus, which
is not committed. When the real corpus lands, create the split once and commit
it deliberately (`git add -f data/splits.json`) so every later run uses the
same documents. `make-split` refuses to overwrite an existing split.
| `show-config` | Print the fully resolved configuration |

Direction is set with `--source` and `--target`, or by an experiment overlay:

```bash
uv run mats-stod translate --strategy B2 --source ta --target si
uv run mats-stod translate --experiment configs/experiments/B2_sliding_window_ta-si.yaml
```

## The three baselines

| Strategy | Context given to each segment | Calls per document |
|---|---|---|
| `B0_full_document` | none, the whole document is one prompt | 1, or one per chunk if it exceeds the token limit |
| `B1_isolated` | none | one per paragraph |
| `B2_sliding_window` | previous `k` paragraphs and their translations | one per paragraph |

`k` is `translation.window_k`, default 3. A B0 run that had to fall back to
chunks records that fact in its report, because a chunked B0 is no longer B0.

## Segmentation

`StructuralSegmenter` is rules only: one segment per block, short blocks merged
forward, long blocks split at sentence boundaries. Zero LLM calls.

`GraftDiscourseSegmenter` reproduces GRAFT's discourse agent. A discourse
starts at one sentence and grows greedily, asking the model once per candidate
next sentence whether it belongs. Cost is one call per boundary considered,
linear in sentences.

The model only ever answers yes or no. Segments are assembled from sentence
spans computed from the source string, so the model cannot rewrite the
document and the invariants hold without trusting it.

The sentence splitter handles Sinhala and Tamil punctuation and does not split
on abbreviations (`කි.මී.`, `எ.கா.`), decimals (`115.5`), or clause numbering
(`3.1`).

## Gold annotation

```bash
uv run mats-stod annotate-template data/samples/circular_01 --out data/gold/templates
# mark segment starts with B in column 2, add edges under EDGES
uv run mats-stod annotate-import data/gold/templates/circular_01.si.annot.tsv
uv run mats-stod segment --segmenter graft   # now reports P/R/F1, Pk, WindowDiff
```

Annotate the Sinhala file and the Tamil file separately: each is a source
document in one direction, with its own segmentation. Gold files are named
`<doc_id>.<lang>.json`, and a run only uses gold matching its source language,
so Sinhala gold can never be scored against a Tamil segmentation.

## Invariants

Enforced by `DiscourseGraph.validate_graph()` on every document at run time,
not only in tests:

1. Segments are non-overlapping and ordered, and concatenating them by offsets
   reproduces `raw_text` apart from inter-segment whitespace.
2. Every edge runs forward in reading order, which makes the graph acyclic by
   construction. Backward dependencies are recorded in `forward_refs`, not as
   edges.
3. No self loops, no duplicate `(src, dst, type)` triples, every endpoint
   exists, every type is in the configured vocabulary.

All text is NFC-normalised once at load. Zero-width joiners are preserved; a
test proves one survives the whole pipeline, because U+200D is letter-forming
in Sinhala and dropping it from ශ්‍රී produces a different word.

## Known limitations

- **Layout extraction is deferred.** Only plain text is parsed, so every block
  is a paragraph, `heading_path` is always empty, and the deterministic
  structural-edge pass produces nothing. The discourse graph will come entirely
  from GRAFT's edges until a layout parser lands. The structural-edge code is
  written and tested against hand-built blocks so it activates unchanged.
- **BLEU is close to meaningless on short segments.** It needs 4-grams, and
  sacrebleu has no Sinhala or Tamil tokeniser. chrF++ is the primary metric for
  this reason.
- **No real corpus yet.** `data/samples/` holds three hand-written synthetic
  documents, committed so tests and demos run. They are not government text and
  must never be reported as results.
- **No learned metric.** A `LearnedMetric` protocol exists; COMET is not
  installed, because it needs a model this machine cannot host and is not
  validated for Sinhala–Tamil.
- **GRAFT's memory agent is not implemented**, and neither are entity or
  terminology handling, the evaluation and correction agent, or Stage 3 review
  and document writing. Each has an interface stub.
- **The GRAFT edge agent is quadratic.** M3 will need the per-document call
  ceiling (`graph.max_pairwise_calls_per_doc`) set against real document
  lengths before it runs on the real corpus.

## Layout of the repository

```
configs/            default.yaml plus one overlay per condition and direction
data/samples/       three synthetic si/ta pairs, committed, used by tests
data/parallel/      the real corpus (not committed)
data/gold/          hand-annotated segmentation and edges (not committed)
src/mats_stod/
  schemas.py        Document, Segment, Edge, DiscourseGraph, TranslationRecord
  config.py         typed settings; no magic constants anywhere else
  io/               loaders, NFC normalisation, run directories, fixed split
  llm/              LLMClient, Gemini provider, cache, cost ledger, FakeLLM
  prompts/          versioned .jinja templates, never inline strings
  parsing/          LayoutParser interface, PlainTextParser, deferred stubs
  segmentation/     sentence splitter, structural and GRAFT segmenters
  translation/      context strategies, translator, protected-content checks
  evaluation/       metrics, segmentation scorer, reports, annotation helper
  pipelines/        baseline, comparison, segmentation
runs/               run outputs (not committed)
```

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
| M0 | Scaffold, schemas, LLM cache, call ledger, metrics, reports | Done |
| M1 | DAG-context translation (D1), orchestrated with LangGraph | Done |
| M2 | Segmentation: GRAFT discourse agent | Done |
| M3 | Edge inference (GRAFT) and DAG assembly | Done |
| M4 | An ablation comparing `dag_context_depth` and graph settings | Not started |

Both translation directions are supported everywhere: `si→ta` and `ta→si`.

## Setup: Windows, macOS and Linux

Follow steps 1–4 for an offline setup. For real Gemini translations through
**Google Cloud Vertex AI**, continue with steps 5–6.

Use **PowerShell on Windows** and **Terminal with bash or zsh on macOS/Linux**.
Commands marked **All platforms** work in either shell. Run commands from the
repository folder unless a step says otherwise.

### 1. Install Git and uv

You need Git and [uv](https://docs.astral.sh/uv/). uv downloads the pinned
Python 3.11 version and manages the virtual environment for you; a separate
Python installation is not required. Installation needs an internet connection.

**Windows — PowerShell**

```powershell
winget install --id Git.Git -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS — Terminal**

Install Apple's command-line tools if Git is not already installed, then install uv:

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Linux — Terminal**

Install Git and curl using your distribution's package manager. For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git curl
```

Then install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal so the new tools are on `PATH`, then check:

```text
git --version
uv --version
```

### 2. Get the code

The setup in [README_WINDOWS.md](README_WINDOWS.md) targets the **`dev` branch**;
use the same branch on all platforms.

**Windows — PowerShell**

Use a short path outside OneDrive, which can lock virtual-environment files:

```powershell
New-Item -ItemType Directory -Force C:\dev
Set-Location C:\dev
git clone -b dev https://github.com/thusykanna/MATS-STOD.git
Set-Location MATS-STOD
```

**macOS / Linux — Terminal**

```bash
mkdir -p ~/dev
cd ~/dev
git clone -b dev https://github.com/thusykanna/MATS-STOD.git
cd MATS-STOD
```

If you already have this checkout, open a terminal in its root folder instead.

### 3. Install dependencies and check the setup

**All platforms — choose one installation:**

```text
uv sync --dev --extra gemini
```

This installs the application, test tools and Google SDK for real translations.
For an offline-only setup without the Google SDK, use `uv sync --dev` instead.

When using Gemini, **keep `--extra gemini` on subsequent `uv sync` commands**;
otherwise uv removes the optional Google SDK.

Run the tests:

```text
uv run pytest
```

The test suite blocks network calls and requires no Google account or API key.
You do not need to activate `.venv`: `uv run` selects the project environment.

### 4. Run the offline demo

**All platforms**

```text
uv run mats-stod make-split --data data/samples
uv run mats-stod compare --provider fake --data data/samples --portion all
```

Create the split only once; `make-split` refuses to overwrite an existing split.
If it already exists, skip that command. The fake provider uses scripted
responses, so this checks the pipeline without making real model calls.

Outputs land in `runs/<run_id>/`:

| File | Contents |
|---|---|
| `report.md` | Scores, call counts and cache hits |
| `results.json` | Machine-readable results |
| `config_used.yaml` | Configuration used for the run |
| `documents/<doc_id>/translation.txt` | Translated text |
| `log.jsonl` | Event log |

The three sample document pairs are synthetic and must not be reported as
research results.

### 5. Configure Vertex AI (optional)

If you chose the offline-only installation, first run:

```text
uv sync --dev --extra gemini
```

Create your local configuration file using the command for your shell.
**If `.env` already exists, edit it instead of overwriting it.**

**Windows — PowerShell**

```powershell
Copy-Item .env.example .env
notepad .env
```

**macOS / Linux — Terminal**

```bash
cp .env.example .env
nano .env
```

You can use any text editor. `.env` is gitignored and loaded automatically;
environment variables already set in your shell take precedence.

You need a Google account and a Google Cloud project with **billing enabled**.
Free-trial credit can be used through a billing-enabled project.

**Install the Google Cloud CLI**

Windows — run the installer and accept its defaults. You can skip `gcloud init`
at the end because the commands below configure the CLI:

```powershell
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& "$env:Temp\GoogleCloudSDKInstaller.exe"
```

macOS — if you use Homebrew:

```bash
brew install --cask google-cloud-sdk
```

Linux, or macOS without Homebrew — follow the
[Google Cloud CLI installation guide](https://cloud.google.com/sdk/docs/install)
for your operating system.

Reopen your terminal, return to the repository folder and run `gcloud --version`.

**Sign in and configure your project — all platforms**

Replace `YOUR_PROJECT_ID` with your project **ID**, not its display name. Create
a project in the [Google Cloud Console](https://console.cloud.google.com) if needed.

First, sign in to the CLI (a browser opens), list your projects and select one:

```text
gcloud auth login
gcloud projects list
gcloud config set project YOUR_PROJECT_ID
```

Check billing. The output should include `billingEnabled: true`; otherwise,
link a billing account under **Billing** in the Cloud Console.

```text
gcloud billing projects describe YOUR_PROJECT_ID
```

Enable Vertex AI, then create Application Default Credentials for the Python
SDK (a second browser login) and set their quota project:

```text
gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

The two logins serve different purposes: `gcloud auth login` authorises CLI
commands; `gcloud auth application-default login` authorises the Python SDK.
Complete both, including the quota-project step.

For an organisation or university project, ask its administrator for the
**Vertex AI User** role (`roles/aiplatform.user`) if your account lacks access.

**Update `.env` — all platforms**

```ini
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
GOOGLE_CLOUD_LOCATION=us-central1
```

No API key is needed for this setup. `configs/default.yaml` already selects `llm.backend: vertex` and `llm.model: gemini-2.5-flash`.

### 6. Verify and run a real translation

**All platforms**

Check configuration first. This makes no model call and costs nothing:

```text
uv run mats-stod check-llm
```

Then test one real call, which uses a small number of tokens and may incur charges:

```text
uv run mats-stod check-llm --send
```

A successful reply confirms that Vertex AI credentials work and the configured
model is reachable.

Start with one sample document:

```text
uv run mats-stod translate --data data/samples --portion all --max-docs 1
```

The default direction is Sinhala to Tamil. To reverse it:

```text
uv run mats-stod translate --data data/samples --portion all --source ta --target si --max-docs 1
```

Open the files in `runs/<run_id>/` to inspect the results. If Sinhala or Tamil
appears as boxes in your terminal, open the UTF-8 output in an editor with a
font that supports those scripts.

## Controlling cost

For pipeline commands such as `translate` and `compare`:

- `--max-docs N` caps the number of documents processed. Start with `1`.
- `--provider fake` uses scripted responses without real model calls.
- `--dry-run` serves cached responses and stops at the first uncached call.

Model responses are cached in `.cache/llm_cache.sqlite`. Identical cached
requests can be reused; changes to prompts or model settings can require new
calls. Reports show call counts and cache hits.

## Troubleshooting

| What you see | What to do |
|---|---|
| `uv`, `git` or `gcloud` is not found | Reopen your terminal after installation and check the tool is on `PATH`. |
| PowerShell says scripts are disabled | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then reopen PowerShell. On a managed machine, follow your organisation's policy. |
| Windows install fails with locked files or access denied | Keep the checkout outside OneDrive, for example in `C:\dev`. |
| `You do not currently have an active account selected` | Run `gcloud auth login`. |
| `google-genai is not installed` | Run `uv sync --dev --extra gemini`. |
| `could not automatically determine credentials` | Run `gcloud auth application-default login`. |
| `Vertex backend needs a project` | Set `GOOGLE_CLOUD_PROJECT` in `.env` and run `check-llm`. |
| `403 SERVICE_DISABLED` | Run `gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID`. |
| `403 PERMISSION_DENIED` | Check the project ID, billing and your account's permissions. |
| `404 ... was not found or your project does not have access` | Check model availability and update `llm.model` or the configured region. |
| `429 RESOURCE_EXHAUSTED` | Wait for quota to recover or reduce the workload with `--max-docs`. |
| gcloud warns about the quota project | Run `gcloud auth application-default set-quota-project YOUR_PROJECT_ID`. |
| `make-split` refuses to overwrite a split | Skip it if the existing split is the one you intend to use. |

Non-retryable configuration errors stop immediately; quota and transient server
errors back off and retry. Model availability can change: use `check-llm --send`
to verify access to the configured model and region.

## Commands

| Command | Purpose |
|---|---|
| `translate` | Translate under the DAG-context condition (D1) |
| `compare` | Run the DAG-context condition, write a results table |
| `segment` | Segment with GRAFT and score against gold |
| `build-graph` | Build the discourse graph with GRAFT, score against gold |
| `eval --hyp DIR --ref DIR` | Score two directories of `.txt` files |
| `annotate-template DOC` | Write a hand-editable gold annotation file |
| `annotate-import FILE` | Convert a filled-in template into gold JSON |
| `make-split` | Create the fixed dev/test split, once |
| `show-config` | Print the fully resolved configuration |
| `check-llm` | Diagnose credentials; `--send` tests one real call |

`data/splits.json` is gitignored because it is derived from the corpus, which
is not committed. When the real corpus lands, create the split once and commit
it deliberately (`git add -f data/splits.json`) so every later run uses the
same documents. `make-split` refuses to overwrite an existing split.

Direction is set with `--source` and `--target`, or by an experiment overlay:

```bash
uv run mats-stod translate --source ta --target si
uv run mats-stod translate --experiment configs/experiments/cap_4.yaml
```

## The DAG-context condition (D1)

`translate` and `compare` run one condition: `D1_dag_context`. A document is
segmented with GRAFT, its discourse graph is built exactly as `build-graph`
builds it, and translation then runs as a LangGraph state graph whose nodes
and edges mirror that discourse graph one for one: a segment's node fires
only once every segment it depends on has already been translated, and
receives those translations as its context (`translation.dag_context_depth`
hops up the graph's parent edges, budgeted by `translation.context_token_budget`).
Segments with no dependency relationship between them fall into the same
LangGraph superstep and translate concurrently.

LangGraph is used only here (DECISIONS.md D25): progress is checkpointed to
`checkpoints.sqlite` inside the run directory, keyed by document, so a run
interrupted partway through resumes from its last completed segment when
re-invoked with the same `--run-id` rather than re-translating the document.

An ablation over `dag_context_depth`, `graph.max_parents` and
`graph.transitive_reduction` is the natural next comparison and is not part
of this stage (see M4 in Status).

## Segmentation

`GraftDiscourseSegmenter` reproduces GRAFT's discourse agent, the only
segmenter this project runs for now. A discourse
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
uv run mats-stod segment   # now reports P/R/F1, Pk, WindowDiff
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
  llm/              LLMClient, Gemini provider, cache, call ledger, FakeLLM
  prompts/          versioned .jinja templates, never inline strings
  parsing/          LayoutParser interface, PlainTextParser, deferred stubs
  segmentation/     sentence splitter, GRAFT discourse segmenter
  graph/            structural + GRAFT pairwise edges, assembly, export
  translation/      graph-derived context, translator, line-break masking, protected-content checks
  evaluation/       metrics, segmentation scorer, reports, annotation helper
  pipelines/        DAG translation (LangGraph), graph building, segmentation, comparison
runs/               run outputs (not committed)
```

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
| M1 | GRAFT memory translation, checkpointed with LangGraph | Done |
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
uv sync --dev --extra gemini --extra comet
```

This installs the application, test tools, Google SDK and COMET for real translations.
For an offline-only setup without the Google SDK, use `uv sync --dev` instead.

For referenced translations, keep both `--extra gemini` and `--extra comet` on
subsequent `uv sync` commands; otherwise uv removes those optional dependencies.

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

The four sample document pairs are synthetic and must not be reported as
research results.

### 5. Configure Vertex AI (optional)

If you chose the offline-only installation, first run:

```text
uv sync --dev --extra gemini --extra comet
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

Run the complete pipeline over every usable folder in `data/parallel`:

```text
uv run mats-stod translate
```

The command asks for the direction using a numbered menu:

```text
1. Sinhala → Tamil
2. Tamil → Sinhala
```

Limit the run to the first usable folder with `--max-docs 1`. Explicit flags
bypass the menu for scripts and repeatable experiments:

```text
uv run mats-stod translate --data data/samples --portion all --source ta --target si --max-docs 1
```

Each document folder needs a file for the selected source language, such as
`notice.si` or `notice.ta`. A target-language file in the same folder is
optional: when present it is used to report chrF++ and BLEU; when absent the
translation is still written and evaluation is marked unavailable. Output
paths are printed after every document.

Translation defaults to `--portion all`. To select `--portion dev` or
`--portion test`, first create the configured split file with `mats-stod make-split`.
A missing split file or unsupported portion is an error; translation does not
silently fall back to the full corpus. In mixed batches, reference-based metrics
(including optional COMET) score only documents with target references.

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
| `translate` | Translate with GRAFT memory (default) or the configured raw-context condition |
| `compare` | Run the DAG-context condition, write a results table |
| `compare-terminology RUN...` | Validate and compare completed terminology experiment runs |
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
uv run mats-stod translate --experiment configs/experiments/raw_context.yaml
```

## GRAFT memory baseline

`translate` and `compare` default to `graft_baseline`. After segmentation and
graph construction, a terminology prepass completes for every discourse before
translation starts. Each discourse is then translated in reading order and its
local memory is extracted. The next discourse uses memories from its direct
graph predecessors. Runtime reference text is never read by the translation
pipeline.

The terminology prepass is enabled by default for `graft_baseline`. Its configurable
methods are `llm_exact` (E0), `python_scan` (E1), `llm_lookup_form` (E2), and
`hybrid` (E3, the default). LLM methods copy a source surface and propose a
source-language lookup form; Python remains authoritative by accepting only forms in
`data/glossaries/dummy_government.si-ta.json`. E0 ignores the proposed lookup form,
E1 makes no terminology LLM call, E2 verifies it only after surface lookup fails, and
E3 combines E2 with the literal glossary scan. Only approved matches enter the
translation prompt. No stemming, lemmatization or morphology generation is performed.
Target inflection is permitted during translation, and glossary terms take precedence
over conflicting GRAFT memory.

Select a method with an experiment overlay:

```bash
uv run mats-stod translate --experiment configs/experiments/terminology_e0_llm_exact.yaml --source si --target ta --portion test --run-id term-e0-si-ta
uv run mats-stod translate --experiment configs/experiments/terminology_e1_python_scan.yaml --source si --target ta --portion test --run-id term-e1-si-ta
uv run mats-stod translate --experiment configs/experiments/terminology_e2_llm_lookup_form.yaml --source si --target ta --portion test --run-id term-e2-si-ta
uv run mats-stod translate --experiment configs/experiments/terminology_e3_hybrid.yaml --source si --target ta --portion test --run-id term-e3-si-ta
```

If independently annotated files exist under `data/gold/terminology/`, runs also
report terminology identification, resolution, recovery and target-realization
metrics. Compare completed compatible runs without rerunning them:

```bash
uv run mats-stod compare-terminology runs/term-e0-si-ta runs/term-e1-si-ta runs/term-e2-si-ta runs/term-e3-si-ta --out terminology-comparison-si-ta
```

The bundled glossary is non-authoritative demonstration data. Version
`dummy-government-v2-gazette` has 37 entries and includes pairs curated from
the `testing_01` parallel reference specifically for workflow verification.
Testing it on that same document is therefore reference leakage and must not be
reported as an unbiased quality result. Use
`configs/experiments/baseline_no_terminology.yaml` for the baseline ablation;
the two raw-context overlays disable the terminology prepass.

The Memory Agent returns five validated components: noun–pronoun mappings,
source/target entity mappings, phrase mappings, discourse connective mappings,
and a short target-language summary. One structured request extracts all five
from the source discourse and its completed translation.

The baseline retains all approved forward edges and automatic neighbor links:
`max_parents: 0`, `max_pair_distance: null`, `transitive_reduction: false`.
Conflicting graph settings are rejected, not silently relabeled as a baseline.
Discourse growth has no character cap by default. Sentence-span preservation,
Sinhala/Tamil sentence rules, and official-document prompts remain adaptations.

Direct-parent memories are merged in reading order. Exact source keys are
deduplicated within each component; the earliest value wins a conflict.
Every retained mapping identifies its originating segment. All parent summaries
are retained. Conflicting later values are recorded outside the prompt for audit.
This exact-key merge is our documented interpretation of earlier-memory priority.

`dag_context_depth` and `context_token_budget` apply only to `dag_raw_context`.
Baseline memory is never silently truncated. `memory.request_token_limit` is an
operational guard on estimated prompt tokens plus reserved output, for translation
and memory requests. It uses the configured character ratio, excludes API/schema
overhead, and is not an exact model tokenizer. Leave headroom; oversized requests
fail explicitly. Invalid memory and unparseable baseline translations also fail
after bounded retries rather than continuing with empty memory or raw output.

Before edge requests, the CLI shows the document's planned decision count.
Plans are counted before enumeration and streamed; distance-bounded plans never
allocate all pairs. The default 4,000-decision ceiling stops a 100-segment
all-pairs document (4,851 decisions) before edge requests, although segmentation
has already run. The batch fails fast: it does not silently exclude documents
or publish an aggregate over an incomplete batch. Earlier artifacts may remain.
Raise the ceiling explicitly to continue the baseline; a distance bound is an
ablation and requires selecting the raw-context condition.

LangGraph checkpoints terminology, translation and memory extraction as separate stages in
`checkpoints.sqlite`. If memory extraction fails, the completed translation is
reused on resume. Schema-invalid terminology, translation, or memory responses
are evicted from the cache, so they cannot permanently block retries. Reuse the same `--run-id` to
resume unchanged work. Fingerprints validate source text, settings, provider,
model, prompt contents, glossary SHA-256, execution version, and the rebuilt graph. Legacy or
incompatible checkpoints require a **fresh run ID**. Changed run settings are
rejected before overwriting saved configuration or provenance. Graph rebuilding
still occurs on resume; its model requests can use the cache.

Reports distinguish logical requests, cache hits, completed responses, and
provider adapter calls (including failed adapter invocations). Internal adapter
or SDK retries are not counted separately. A translation parse retry is a new
logical request. Counts describe this invocation, not cumulative checkpoint
history. `calls` in JSON remains the completed-response count for compatibility.

Artifacts include `terminology.json`, optional `terminology_evaluation.json`, the run-level `glossary_snapshot.json`,
`memories.json`, `memory_contexts.json`, `records.json`, and the graph. They preserve
term candidates, approved pairs, unmatched candidates, local memories, merged
context, conflicts, prompts, and provenance. Call reports distinguish terminology
extraction, memory extraction, and translation.

**Migration:** use a fresh run ID; prior raw-context checkpoints are incompatible.
See [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for the code map and
[README_WINDOWS.md](README_WINDOWS.md) for PowerShell examples.

The implementation follows the mechanism in
[GRAFT §3](https://arxiv.org/html/2507.03311v1#S3), not an exact reproduction of
its scores or prompts. The linked reference repository was unavailable during
implementation. JSON schema, combined extraction request, exact-key merging,
and the request ceiling are explicit local choices; see the guide.

For later comparisons, `configs/experiments/direct_raw_context.yaml` uses the
unpruned graph with direct-parent raw passages, while `raw_context.yaml` restores
the prior capped, depth-2 approach. `compare` scores one configured condition per
invocation. Use the same corpus, model, and cached graph decisions with distinct
run IDs. Verify exported graph equality when isolating memory's effect; the
prior raw overlay also changes segmentation limits and parent pruning.

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
- **No real corpus yet.** `data/samples/` holds four hand-written synthetic
  documents, committed so tests and demos run. They are not government text and
  must never be reported as results.
- **COMET requires an optional model download.** It is enabled for referenced
  translation runs and requires the `comet` dependency extra. Its score supplements
  chrF++ and BLEU; it does not replace terminology-specific evaluation.
- **Memory extraction can be wrong.** Schema validation checks structure, not
  semantic correctness. Memory and earlier-value conflict resolution still need
  evaluation on Sinhala–Tamil documents. Automatic correction and document review
  are not implemented.
- **The bundled glossary is only a workflow fixture.** Its 37 pairs include terms
  curated from the `testing_01` reference and are neither independently reviewed nor
  suitable for an unbiased measurement on that document. Lookup supports exact
  preferred/alias forms, literal scanning, and glossary-verified LLM lookup-form
  proposals. Fuzzy search, stemming, lemmatization and morphology generation are not
  implemented.
- **The GRAFT edge agent is quadratic.** M3 will need the per-document call
  ceiling (`graph.max_pairwise_calls_per_doc`) set against real document
  lengths before it runs on the real corpus.

## Layout of the repository

```
configs/            default.yaml plus one overlay per condition and direction
data/samples/       four synthetic si/ta pairs, committed, used by tests
data/glossaries/    local bilingual glossary fixtures
data/parallel/      default translation inputs; one folder per document
data/gold/          hand-annotated segmentation and edges (not committed)
src/mats_stod/
  schemas.py        Document, Segment, Edge, DiscourseGraph, TranslationRecord
  config.py         typed settings; no magic constants anywhere else
  io/               loaders, NFC normalisation, run directories, fixed split
  llm/              LLMClient, Gemini provider, cache, call ledger, FakeLLM
  prompts/          versioned .jinja templates, never inline strings
  terminology/      candidate extraction, glossary lookup and term records
  parsing/          LayoutParser interface and PlainTextParser
  segmentation/     sentence splitter, GRAFT discourse segmenter
  graph/            structural + GRAFT pairwise edges, assembly, export
  translation/      graph-derived context, translator, line-break masking, protected-content checks
  evaluation/       metrics, segmentation scorer, reports, annotation helper
  pipelines/        DAG translation (LangGraph), graph building, segmentation, comparison
runs/               run outputs (not committed)
```

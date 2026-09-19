# Decisions

Every non-obvious choice, with the alternative that was rejected and why. This
is the file to revise before a panel defence.

Format: **Decision** / *Alternative rejected* / Reason.

---

## Method: relationship to GRAFT

MATS-STOD reuses the technique of GRAFT (Dutta, Manchanda, Bapat, Gurjar and
Bhattacharyya, *GRAFT: A Graph-based Flow-aware Agentic Framework for
Document-level Machine Translation*, EMNLP 2025 Industry Track, arXiv
2507.03311). The reference implementation is at
`github.com/himanshu-dutta/graft`.

### D1. GRAFT's discourse agent reproduced faithfully

**Decision.** Segmentation grows a discourse one sentence at a time, asking the
model a yes/no question per candidate sentence, exactly as GRAFT's
`DiscourseAgent` does.
*Alternative rejected.* An LLM that returns merge and split operations over
rule-based segments, which would cost fewer calls.
**Reason.** The project reuses a published technique, so the comparison to that
paper has to be like for like. The operation-based variant is a different
method and would need its own justification.

### D2. GRAFT's pairwise edge agent reproduced faithfully

**Decision.** Every segment is linked to its successor, and every non-adjacent
ordered pair is judged by one yes/no call. Cost is quadratic in segments per
document.
*Alternative rejected.* One call per segment against a capped candidate list,
which is linear and returns typed edges with evidence spans.
**Reason.** Explicit project decision to stay faithful to the published method.
The cost is contained by the disk cache, `--max-docs`, and a per-document call
ceiling (`graph.max_pairwise_calls_per_doc`) that aborts rather than
overspending. The linear variant remains available as future work.

### D3. The model never returns document text

**Decision.** In both the discourse agent and the edge agent the model answers
yes or no. Segments are assembled from sentence spans computed from the source
string.
*Alternative rejected.* Asking the model to return the segmented text, as is
common in prompt-based chunking.
**Reason.** Invariant 1 requires that concatenating segments reproduces the
source apart from inter-segment whitespace. If the model returns text it can
silently normalise, drop a zero-width joiner, or fix a typo, and the invariant
becomes unenforceable.

### D4. Discourses do not grow across atomic layout blocks

**Decision.** Headings, list items, table cells and captions each form their own
group; a discourse never spans two of them.
*Alternative rejected.* GRAFT's flat treatment, where the document is one
stream of sentences.
**Reason.** GRAFT is evaluated on TED talks and novels, which are flat prose.
Government circulars are not: gluing a heading to the paragraph under it
destroys the only structure available. The rule has no effect today because
layout parsing is deferred and every block is a paragraph, but it is in place
for when it lands.

### D5. GRAFT's memory agent is out of scope

**Decision.** The consistency memory (noun-pronoun map, entity map,
connectives, phrase map, translation summary, context summary: six extra LLM
calls per segment) is not implemented. A `ContextEncoder` extension point marks
where it would go.
*Alternative rejected.* Porting it now.
**Reason.** It is outside the declared scope of this build, and it would
roughly septuple the per-segment cost on a free-trial budget.

### D6. `graft_dependency` added to the edge type vocabulary

**Decision.** Edges from the GRAFT edge agent carry the type
`graft_dependency` with a null evidence span.
*Alternative rejected.* Forcing them into one of the six semantic types.
**Reason.** The agent answers one yes/no question and therefore cannot name a
relation. Labelling such an edge `coreference` would be an unsupported claim.
The consequence is that the evidence-span hallucination check does not apply to
these edges, and the report must say so rather than print a zero.

---

## Evaluation

### D7. chrF++ is primary, BLEU secondary

**Decision.** chrF++ (`word_order=2`) is the headline metric; BLEU is reported
with `tokenize=none` and its tokeniser stated.
*Alternative rejected.* BLEU as primary, as in most MT papers.
**Reason.** sacrebleu ships no word tokeniser for Sinhala or Tamil. Both
languages are morphologically rich, so whitespace tokens are large and 4-gram
matches are rare. A character n-gram metric degrades gracefully where word BLEU
depends on a tokeniser that does not exist.

### D8. BLEU returns 0 below four tokens, and that is left visible

**Decision.** No smoothing or floor is applied; a test documents the behaviour.
*Alternative rejected.* Switching on sacrebleu smoothing to avoid zeros.
**Reason.** BLEU needs 4-grams. Silently smoothing would hide the fact that
BLEU is close to meaningless on short Sinhala and Tamil segments, which is the
evidence for D7.

### D9. Documents are the scoring unit

**Decision.** Corpus scores treat each document as one segment, matching the
d-BLEU convention GRAFT reports.
*Alternative rejected.* Sentence-level scoring after realigning output to
reference sentences.
**Reason.** Realignment needs a sentence aligner for Sinhala and Tamil that the
project does not have; GRAFT's own approach of padding or truncating a sentence
list to match introduces an artefact that would be hard to defend.

### D10. A fixed split file, never reshuffled

**Decision.** `data/splits.json` is written once and read forever;
`create_split` refuses to overwrite without an explicit flag.
*Alternative rejected.* Seeding a shuffle at run time.
**Reason.** A split that moves makes every earlier number incomparable, and a
seeded shuffle still moves when the corpus grows.

### D11. Gold files are tagged with their source language

**Decision.** Gold segmentation and gold edges are stored as
`<doc_id>.<lang>.json`, and a run only loads gold whose language matches its
source document.
*Alternative rejected.* Keying gold by document id alone.
**Reason.** Each pair is evaluated in both directions, so the same `doc_id` is
a Sinhala source in one run and a Tamil source in the next. Untagged gold was
silently scored against the wrong side and reported an F1 of zero that looked
like a segmenter failure. A regression test covers it.

---

## Engineering

### D12. One provider interface; vendor SDKs confined to one module each

**Decision.** Everything depends on `LLMClient`. `llm/gemini.py` is the only
module importing the Google SDK; `llm/openai_compat.py` the only one importing
`openai`.
*Alternative rejected.* Calling the SDK where it is needed.
**Reason.** Model choice is still an open research question. Confining the SDK
means swapping models cannot touch pipeline code, and the whole test suite runs
with no cloud dependency installed.

### D13. Gemini on Vertex AI is the default backend

**Decision.** `llm.backend: vertex`, using a GCP project and Application
Default Credentials. AI Studio with an API key is kept as a fallback in the
same module.
*Alternative rejected.* AI Studio as default, which needs only an API key.
**Reason.** The available budget is Google Cloud free-trial credit, which
applies to Vertex. Both backends produce identical cache keys for identical
prompts, so switching never re-spends tokens.

### D14. Cache keyed on provider, model, messages, schema and parameters

**Decision.** SHA-256 over all five, with parameters sorted.
*Alternative rejected.* Keying on the prompt text alone.
**Reason.** A temperature or schema change produces a different answer; a cache
that ignored them would serve a stale response from a different condition and
quietly corrupt an experiment.

### D15. Cost is reported twice: billed and cold

**Decision.** The ledger reports what the run actually spent (cache hits free)
and what it would cost from an empty cache.
*Alternative rejected.* Reporting only actual spend.
**Reason.** The write-up needs the cost of the method, not the cost of the
fifth re-run of it.
**Status update (2026-09-19).** USD cost tracking was removed for the initial
implementation stage; `llm/cost.py` was renamed to `llm/ledger.py` and now
reports only calls, cache hits and token counts. Reinstating a cost figure
means restoring a price table and the billed/cold split described here.

### D16. An unpriced model is reported, not counted as free

**Decision.** A model absent from `llm.prices_usd_per_mtok` contributes zero to
the cost total, and the run records which models those were; the report prints
a warning that the total is incomplete.
*Alternative rejected.* Defaulting an unknown model's price to zero silently,
which is what the code did at first.
**Reason.** Model names change faster than a config file does. A run on a new
model printed "USD spent 0.0" while really spending money, which is a false
statement in a report that will be defended.
**Status update (2026-09-19).** Superseded by the removal described in D15:
with no price table, there is nothing to be unpriced. The underlying risk this
guarded against still applies if pricing returns.

### D17. A dry run raises rather than inventing an answer

**Decision.** `--dry-run` serves cached calls and raises `DryRunExhausted` on
the first uncached one.
*Alternative rejected.* Returning a placeholder so the run completes.
**Reason.** A dry run that completes with invented text produces a report that
looks real. Failing loudly is the only safe behaviour.

### D18. An unparseable yes/no answer counts as a boundary or a missing edge

**Decision.** Parse failures in segmentation and edge inference are counted and
resolved conservatively, without a retry.
*Alternative rejected.* Retrying each failure.
**Reason.** These calls are the volume of the pipeline. A retry policy on them
could double the cost of the whole pass to recover a handful of decisions. The
counts are reported so the rate is visible.

### D19. Translation retries once, then falls back to raw text and flags it

**Decision.** A parse failure retries with an added reminder; a second failure
records the raw text plus `fell_back_to_raw_text`.
*Alternative rejected.* Raising, losing the document.
**Reason.** The retry appends text rather than resending the same prompt,
because an identical prompt would be served from the cache and fail
identically. Failing the whole document over one segment wastes the tokens
already spent on it.

### D20. Protected content is flagged, never repaired

**Decision.** A digit or reference-number divergence sets a flag on the record;
the translation is left as produced.
*Alternative rejected.* Substituting the source value back in.
**Reason.** How often the model corrupts a reference number is a result worth
reporting. A silent repair would erase the measurement and could produce
ungrammatical output.

### D21. NFC normalisation once at load, and zero-width joiners preserved

**Decision.** `io/text.normalise` runs at load and nowhere else. ZWJ (U+200D)
is never stripped, and whitespace-only loss is the only loss invariant 1
tolerates.
*Alternative rejected.* Normalising defensively wherever text is handled.
**Reason.** A second normalisation pass would shift character offsets that
segments and edges already point at. ZWJ is letter-forming in Sinhala: removing
it from ශ්‍රී produces a different word. A test asserts it survives the whole
pipeline.

### D22. Artifacts are deterministic; timestamps live elsewhere

**Decision.** `results.json` and `report.md` contain no timestamps; those go in
`run_meta.json` and `log.jsonl`.
*Alternative rejected.* Stamping every artifact.
**Reason.** Two runs on identical input produce byte-identical artifacts, so a
diff between runs shows only what actually changed. A test asserts this.

### D23. Topological order breaks ties by reading order

**Decision.** Kahn's algorithm with ties resolved by segment order.
*Alternative rejected.* Any valid topological order.
**Reason.** A non-deterministic order changes which translations are in context
for later segments, which changes prompts, which changes cache keys. The run
would stop being reproducible.

### D24. Baselines use their own naive paragraph splitter

**Decision.** B1 and B2 segment on blank lines via `NaiveParagraphSegmenter`,
not via `StructuralSegmenter`.
*Alternative rejected.* Sharing the structural segmenter.
**Reason.** The baselines are the control. If improving structural segmentation
also improved the baseline it is measured against, the comparison would move
under its own feet.
**Status update (2026-09-19).** B1, B2 and `StructuralSegmenter` were removed
to keep the initial implementation stage to GRAFT only; `NaiveParagraphSegmenter`
went with them. Only B0 (whole document) remains. This decision's reasoning
still applies and should be reinstated if B1/B2 return for comparison.
**Status update (2026-09-20).** B0 was also removed (see D32): there is no
baseline condition left in the codebase at all, only D1. This decision's
reasoning is dormant, not void, and applies again to whichever baseline is
reintroduced as the ablation's control.

### D25. LangGraph is not used in Phase A or in segmentation

**Decision.** Baselines and segmentation are plain functions. LangGraph enters
only with the DAG translation pipeline, which has real state and needs
checkpointing.
*Alternative rejected.* Building everything as graphs for uniformity.
**Reason.** A loop over paragraphs has no branching. Wrapping it in a state
machine would add a dependency and explain nothing.
**Status update (2026-09-20).** The DAG translation pipeline now exists
(D32) and follows this decision as written: a `StateGraph` with one node per
segment, wired from the discourse graph's own edges, checkpointed to SQLite.
Segmentation and edge inference remain plain functions, unchanged.

### D26. Sentence splitting is rule-based with an abbreviation lookahead

**Decision.** A full stop is not a boundary when it sits inside a decimal, a
clause number, or a known abbreviation, where an abbreviation-internal stop
only counts if the rest of the abbreviation actually follows.
*Alternative rejected.* Matching any abbreviation prefix.
**Reason.** `කි.` opens the abbreviation `කි.මී.` but is also a complete Sinhala
word. Without the lookahead every occurrence of it swallowed the following
sentence boundary. A test covers exactly this case.

### D27. Development on Python 3.11 installed by uv

**Decision.** `requires-python = ">=3.11,<3.14"`, with the interpreter provided
by `uv python install 3.11`.
*Alternative rejected.* Using the system Python 3.14.
**Reason.** The build constraint names 3.11 and several dependencies lag new
releases; `uv` pins it without touching the system installation.

---

### D29. The edge-inferrer registry names every inferrer before any exists

**Decision.** `build_edge_inferrer` lists all four names (`graft`,
`predecessor`, `tfidf`, `none`) from the first commit. An inferrer that is
named but not yet written raises `NotImplementedError` naming the file to
create.
*Alternative rejected.* Adding each name to the registry as its inferrer
lands.
**Reason.** The inferrers are being written by different people at the same
time. If each one had to add its own branch to the registry, every inferrer
would touch the same function and every merge would conflict on it. Naming
them all up front means a new inferrer is a new file and nothing else. The
error message is part of the contract, so it is covered by a test.
**Status update (2026-09-19).** `predecessor`, `tfidf` and `none` were never
built and were dropped from `graph/edge_inference.py`, along with
`build_edge_inferrer` itself: with only `graft` left, the registry was a
pass-through and `graph_pipeline.py` now constructs `GraftPairwiseEdgeInferrer`
directly. The `EdgeInferrer` interface stays, so a future comparison inferrer
is a new class and one call-site change, not a registry redesign.

### D30. The parent cap keeps the nearest parents by reading order

**Decision.** When a segment has more than `graph.max_parents` parents, the
ones kept are those closest to it in reading order. The rule is named in
`assemble.PARENT_CAP_RULE` and written into every graph's metadata.
*Alternative rejected.* Ranking parents by edge confidence.
**Reason.** GRAFT's edge agent answers one yes/no question and returns no
confidence (D6), so there is no score to rank by. Distance is the only
signal available.
**Consequence, which must be stated in any report:** the cap discards the
most distant parents first, and those are exactly the dependencies a sliding
window cannot supply. A cap set too low therefore removes the effect D1 is
meant to demonstrate before it can be measured. `max_parents` is a config
value so this can be run as an ablation rather than assumed.

### D31. Transitive reduction is off by default

**Decision.** `graph.transitive_reduction` defaults to false; the assembled
graph keeps edges that a longer path already implies.
*Alternative rejected.* Reducing by default for a cleaner graph.
**Reason.** Reduction removes short-range edges whenever a longer path exists
between the same pair. Those short-range edges are the ones a sliding window
would also have supplied, so removing them changes what D1 is being credited
for. It is a legitimate ablation and a genuine help when reading a diagram,
but it is not tidying, so it is not the default.

### D32. B0 is removed; D1 (DAG-context translation) is the only translation condition

**Decision.** `WholeDocumentSegmenter`, `B0FullDocument`, `chunk_document` and
`pipelines/baseline.py` are deleted rather than kept alongside D1.
`pipelines/dag_translate.py` segments with GRAFT, builds the discourse graph
exactly as `build-graph` does, and translates it as a LangGraph `StateGraph`
with one node per segment, wired from the graph's own edges (D25): a node
fires once every segment it depends on has translated, receiving those
translations as context (`translation.dag_context_depth` hops up
`DiscourseGraph.ancestors`, budgeted by `translation.context_token_budget`,
in `translation.context.build_dag_context`). Independent segments fall into
the same superstep and translate concurrently. Progress checkpoints to
`checkpoints.sqlite` in the run directory, keyed by document, so a run
resumed with the same `--run-id` does not re-translate completed segments.
*Alternative rejected.* Keeping B0 running alongside D1 as the ablation's
control, per M4's original framing ("DAG-context translation and the
ablation").
**Reason.** Explicit project decision: a baseline kept only for a future
comparison, with no comparison yet run, is unused code paid for on every
change to `Segment`, `TranslationRecord` or the translation prompt. D24's
reasoning for a baseline stays valid and is recorded there as dormant; it
reactivates once an ablation actually needs a control to run against, at
which point a baseline is added back deliberately rather than dragged along
speculatively.
**Consequence, which must be stated in any report:** there is currently no
in-repo comparison point for whether D1's graph-scoped context beats
translating each segment alone or beats a whole-document prompt. Numbers
from D1 are reported on their own until M4 reintroduces a control.

### D33. Resuming a LangGraph thread requires invoking it with `None`, not a fresh initial state

**Decision.** `dag_translate.translate_document` checks
`checkpointer.get_tuple(config)` before invoking: an existing checkpoint means
`app.invoke(None, config=config)`; none means `app.invoke({"records": {}}, config=config)`.
*Alternative rejected.* Always invoking with `{"records": {}}` and trusting
the checkpointer to merge it against saved state.
**Reason.** It does not: `StateGraph.invoke` treats a non-`None` input as a
new run and starts the thread over, silently discarding the checkpoint and
re-translating every segment, LLM cache or not. This was caught by measuring
call counts across a simulated crash-and-resume in a fixed test
(`test_dag_resumes_from_checkpoint_after_a_crash`) before it could surface as
a quietly expensive re-run in production; the failure mode produces no error,
only extra calls, so nothing but a call-count assertion catches it.

### D28. Layout extraction is deferred

**Decision.** Only `PlainTextParser` exists, splitting on blank lines. Every
block is a paragraph at level 0. Markdown, PDF and DOCX parsers raise
`NotImplementedError` behind the same interface.
*Alternative rejected.* Implementing heading, list and table detection now.
**Reason.** Explicit project decision to defer it.
**Consequence, which must be stated in any report:** `heading_path` is empty on
every segment, and the deterministic structural-edge pass yields no edges. The
discourse graph is therefore built entirely from GRAFT's predecessor and
pairwise edges. The structural-edge code is implemented and unit-tested against
hand-built blocks so that it activates unchanged when a layout parser lands.

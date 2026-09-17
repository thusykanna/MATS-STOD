"""Typed settings loaded from YAML.

Nothing in the pipeline may read a magic constant: every threshold, path,
prompt version and model name arrives through `Settings` so that an experiment
condition is fully described by the YAML that produced it, and that YAML is
copied into the run directory.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .schemas import EDGE_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class LangSettings(BaseModel):
    source: str = "si"
    target: str = "ta"
    names: dict[str, str] = Field(
        default_factory=lambda: {"si": "Sinhala", "ta": "Tamil", "en": "English"}
    )

    @property
    def direction(self) -> str:
        return f"{self.source}-{self.target}"

    @property
    def source_name(self) -> str:
        return self.names.get(self.source, self.source)

    @property
    def target_name(self) -> str:
        return self.names.get(self.target, self.target)

    def flipped(self) -> LangSettings:
        return LangSettings(source=self.target, target=self.source, names=self.names)


class LLMSettings(BaseModel):
    provider: str = "gemini"
    backend: str = "vertex"  # vertex | ai_studio
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    top_p: float | None = None
    timeout_s: float = 120.0
    max_retries: int = 3
    #: Gemini 2.5+ reasons before answering and charges those tokens against
    #: max_output_tokens. None leaves the model's default; 0 disables it.
    #: Call sites may override per call.
    thinking_budget: int | None = None
    cache_path: str = ".cache/llm_cache.sqlite"
    use_cache: bool = True
    # Vertex specifics; values normally come from the environment, these are
    # only fallbacks so a config can pin a project for reproducibility.
    project: str | None = None
    location: str = "us-central1"
    #: USD per million tokens. Used for the cost ledger only; wrong numbers
    #: make the ledger wrong, never the translation. A model absent from this
    #: table is reported as unpriced, not as free.
    prices_usd_per_mtok: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
            "fake": {"input": 0.0, "output": 0.0},
        }
    )

    def price_for(self, model: str) -> dict[str, float]:
        return self.prices_usd_per_mtok.get(model, {"input": 0.0, "output": 0.0})


class SentenceSplitSettings(BaseModel):
    terminators: list[str] = Field(default_factory=lambda: [".", "?", "!", "।", "॥", "…"])
    #: Tokens that end in a terminator but do not end a sentence.
    abbreviations: list[str] = Field(
        default_factory=lambda: [
            "කි.මී.",
            "අ.පො.ස.",
            "ව.ප.",
            "අංක.",
            "රු.",
            "எ.கா.",
            "கி.மீ.",
            "அ.பொ.த.",
            "எண்.",
            "ரூ.",
            "Dr.",
            "Mr.",
            "Mrs.",
            "Ms.",
            "No.",
            "Rs.",
            "etc.",
        ]
    )
    min_sentence_chars: int = 2
    #: A line break inside a block ends a sentence when the line is at
    #: most this long; longer lines are treated as hard wrapping.
    newline_break_max_chars: int = 120


class SegmentationSettings(BaseModel):
    segmenter: str = "structural"  # structural | graft
    min_segment_chars: int = 40
    max_segment_chars: int = 1200
    #: GRAFT's max_discourse_length analogue: a hard cap that forces a
    #: boundary regardless of what the LLM answers, so one runaway "yes" chain
    #: cannot swallow a document.
    max_discourse_chars: int = 2048
    merge_short_blocks: bool = True
    split_long_blocks: bool = True
    #: Segment types that are never grown across by the GRAFT segmenter.
    atomic_types: list[str] = Field(
        default_factory=lambda: ["heading", "list_item", "table_cell", "caption"]
    )
    sentences: SentenceSplitSettings = Field(default_factory=SentenceSplitSettings)
    prompt_version: str = "discourse_v1"
    boundary_tolerance_chars: int = 2
    #: Output cap for one yes/no decision. Small, but not so small that a
    #: model which prefixes its answer gets truncated.
    decision_max_tokens: int = 16
    #: 0 disables reasoning for the yes/no call. Without this a thinking model
    #: spends the whole output budget reasoning and returns nothing, which
    #: would silently turn every decision into a segment boundary. Raise it to
    #: study whether reasoning improves boundary quality.
    decision_thinking_budget: int | None = 0


class GraphSettings(BaseModel):
    edge_inferrer: str = "graft"  # graft | predecessor | tfidf | none
    edge_types: list[str] = Field(default_factory=lambda: list(EDGE_TYPES))
    max_parents: int = 4
    #: GRAFT's edge agent is quadratic. This is the per-document ceiling on
    #: pairwise calls; exceeding it aborts rather than silently spending the
    #: token budget.
    max_pairwise_calls_per_doc: int = 4000
    #: null keeps GRAFT faithful (every earlier segment is a candidate).
    max_pair_distance: int | None = None
    tfidf_threshold: float = 0.8
    transitive_reduction: bool = False
    prompt_version: str = "edge_v1"

    @field_validator("edge_types")
    @classmethod
    def _known_types(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in EDGE_TYPES]
        if unknown:
            raise ValueError(f"unknown edge types in config: {unknown}")
        return v


class ProtectSettings(BaseModel):
    """Patterns whose content must survive translation unchanged."""

    enabled: bool = True
    patterns: list[str] = Field(
        default_factory=lambda: [
            r"\d[\d,./-]*",  # numbers, dates, reference numbers
            r"[A-Z]{2,}(?:/[A-Z0-9]+)+",  # MOF/2023/145
        ]
    )
    check_digits: bool = True


class TranslationSettings(BaseModel):
    strategy: str = "B1_isolated"
    prompt_version: str = "translate_v1"
    domain_note: str = "Official government document. Formal register."
    window_k: int = 3
    #: B0 falls back to fixed chunks above this; recorded when it happens.
    full_doc_token_limit: int = 12000
    chunk_token_size: int = 4000
    #: Rough characters-per-token for Sinhala/Tamil used only for budgeting
    #: decisions, never for reported token counts.
    chars_per_token_estimate: float = 2.5
    context_token_budget: int = 2000
    dag_context_depth: int = 2
    retry_on_parse_failure: int = 1
    protect: ProtectSettings = Field(default_factory=ProtectSettings)


class EvaluationSettings(BaseModel):
    primary_metric: str = "chrf++"
    #: sacrebleu has no Sinhala or Tamil word tokeniser; see DECISIONS.md.
    bleu_tokenize: str = "none"
    chrf_word_order: int = 2
    chrf_char_order: int = 6
    chrf_beta: int = 2
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 12345


class PathSettings(BaseModel):
    parallel: str = "data/parallel"
    samples: str = "data/samples"
    gold_segmentation: str = "data/gold/segmentation"
    gold_edges: str = "data/gold/edges"
    runs: str = "runs"
    split_file: str = "data/splits.json"


class Settings(BaseModel):
    """The whole configuration of one run."""

    run_name: str = "default"
    seed: int = 12345
    langs: LangSettings = Field(default_factory=LangSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True, allow_unicode=True)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def load_settings(
    config_path: str | Path | None = None,
    overlays: list[str | Path] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load default.yaml, then experiment overlays, then explicit overrides.

    Later sources win. Overlays exist so an experiment condition is a small
    diff against the default rather than a copy that silently drifts.
    """
    base_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data = _read_yaml(base_path)
    for overlay in overlays or []:
        data = _deep_merge(data, _read_yaml(Path(overlay)))
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.model_validate(data)

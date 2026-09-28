"""COMET, an optional learned metric (needs the `comet` extra).

Scores one number per document: the mean COMET score over paragraphs when the
source, hypothesis and reference have the same number of paragraphs, otherwise
over the whole document as a single unit (long documents may be truncated by
the model in that case).
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class CometMetric:
    """Implements the LearnedMetric protocol from metrics.py."""

    name = "comet"

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 8,
        gpus: int = 0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus  # 0 = CPU
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from comet import download_model, load_from_checkpoint
            except ModuleNotFoundError as exc:
                if exc.name != "comet":
                    raise
                raise RuntimeError(
                    "COMET is not installed. Run: uv sync --dev --extra gemini --extra comet"
                ) from exc
            self._model = load_from_checkpoint(download_model(self.model_name))
        return self._model

    def score(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> list[float]:
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        output = self._load().predict(
            data, batch_size=self.batch_size, gpus=self.gpus, progress_bar=False
        )
        return [float(x) for x in output.scores]


def paragraph_units(src: str, hyp: str, ref: str) -> tuple[list[str], list[str], list[str]]:
    """Split into aligned paragraphs, or fall back to the whole document."""
    parts = [
        [p.strip() for p in _PARAGRAPH_BREAK.split(text.strip()) if p.strip()]
        for text in (src, hyp, ref)
    ]
    s, h, r = parts
    if s and len(s) == len(h) == len(r):
        return s, h, r
    return [src.strip()], [hyp.strip()], [ref.strip()]


def score_documents(
    metric: CometMetric | Any,
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
) -> list[float]:
    """One COMET score per document."""
    units_s: list[str] = []
    units_h: list[str] = []
    units_r: list[str] = []
    owner: list[int] = []
    for i, (s, h, r) in enumerate(zip(sources, hypotheses, references)):
        us, uh, ur = paragraph_units(s, h, r)
        units_s += us
        units_h += uh
        units_r += ur
        owner += [i] * len(us)

    scores = metric.score(units_s, units_h, units_r)
    per_doc: list[list[float]] = [[] for _ in sources]
    for i, sc in zip(owner, scores):
        per_doc[i].append(sc)
    return [round(mean(x), 4) for x in per_doc]


def add_comet(
    doc_scores: list[Any],
    corpus: Any,
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    metric: CometMetric | Any,
) -> None:
    """Store COMET in DocScore.extra and CorpusScore.extra (key: "comet")."""
    scores = score_documents(metric, sources, hypotheses, references)
    for doc, sc in zip(doc_scores, scores):
        doc.extra["comet"] = sc
    corpus.extra["comet"] = round(mean(scores), 4)



def add_comet_if_enabled(
    evaluation: Any,
    doc_scores: list[Any],
    corpus: Any,
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
) -> None:
    """No-op unless evaluation.comet_enabled is true."""
    if not evaluation.comet_enabled:
        return
    metric = CometMetric(
        evaluation.comet_model, evaluation.comet_batch_size, evaluation.comet_gpus
    )
    add_comet(doc_scores, corpus, sources, hypotheses, references, metric)
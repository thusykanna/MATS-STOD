"""Translation quality metrics.

chrF++ is primary because sacrebleu ships no word tokeniser for Sinhala or
Tamil, and both are morphologically rich: a character n-gram metric degrades
gracefully where a word-BLEU depends on a tokeniser that does not exist.
BLEU is reported as a secondary number with its tokeniser stated, so it can be
compared against published work rather than trusted on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import sacrebleu

from ..config import EvaluationSettings


@dataclass
class DocScore:
    doc_id: str
    chrf: float
    bleu: float
    n_chars_hyp: int
    n_chars_ref: int
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class CorpusScore:
    chrf: float
    bleu: float
    n_docs: int
    extra: dict[str, float] = field(default_factory=dict)


class LearnedMetric(Protocol):
    """Extension point for COMET or a fine-tuned estimator.

    Declared now so the report writer has a stable place to put the column,
    but deliberately not installed: COMET needs a GPU-sized download that this
    machine cannot host, and an unvalidated learned metric for Sinhala-Tamil
    would be worse than no number.
    """

    name: str

    def score(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ) -> list[float]:
        ...


def _chrf(settings: EvaluationSettings) -> sacrebleu.metrics.CHRF:
    return sacrebleu.metrics.CHRF(
        char_order=settings.chrf_char_order,
        word_order=settings.chrf_word_order,
        beta=settings.chrf_beta,
    )


def _bleu(settings: EvaluationSettings) -> sacrebleu.metrics.BLEU:
    return sacrebleu.metrics.BLEU(tokenize=settings.bleu_tokenize)


def score_document(
    doc_id: str, hypothesis: str, reference: str, settings: EvaluationSettings
) -> DocScore:
    """Score one document as a single segment (document-level scoring)."""
    hyp = hypothesis.strip()
    ref = reference.strip()
    chrf = _chrf(settings).corpus_score([hyp], [[ref]]).score
    bleu = _bleu(settings).corpus_score([hyp], [[ref]]).score
    return DocScore(
        doc_id=doc_id,
        chrf=round(chrf, 4),
        bleu=round(bleu, 4),
        n_chars_hyp=len(hyp),
        n_chars_ref=len(ref),
    )


def score_corpus(
    hypotheses: list[str], references: list[str], settings: EvaluationSettings
) -> CorpusScore:
    """Corpus-level score over whole documents.

    Documents are the units, which is what d-BLEU means in the document-level
    MT literature and what GRAFT reports.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references differ in length")
    hyps = [h.strip() for h in hypotheses]
    refs = [[r.strip() for r in references]]
    chrf = _chrf(settings).corpus_score(hyps, refs).score
    bleu = _bleu(settings).corpus_score(hyps, refs).score
    return CorpusScore(chrf=round(chrf, 4), bleu=round(bleu, 4), n_docs=len(hyps))


def paired_bootstrap(
    baseline: list[str],
    system: list[str],
    references: list[str],
    settings: EvaluationSettings,
    metric: str = "chrf",
) -> dict[str, Any]:
    """Paired bootstrap resampling between two systems on the same documents.

    Reported so that a small chrF++ gain is not presented as a result when the
    document count is too small to support it.
    """
    import random

    if not (len(baseline) == len(system) == len(references)):
        raise ValueError("all three lists must have the same length")
    n = len(references)
    if n == 0:
        raise ValueError("no documents to compare")

    scorer = _chrf(settings) if metric == "chrf" else _bleu(settings)

    def corpus(hyps: list[str], refs: list[str]) -> float:
        return scorer.corpus_score(hyps, [refs]).score

    base_score = corpus(baseline, references)
    sys_score = corpus(system, references)
    observed = sys_score - base_score

    rng = random.Random(settings.bootstrap_seed)
    wins = 0
    deltas: list[float] = []
    for _ in range(settings.bootstrap_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        b = corpus([baseline[i] for i in idx], [references[i] for i in idx])
        s = corpus([system[i] for i in idx], [references[i] for i in idx])
        delta = s - b
        deltas.append(delta)
        if delta > 0:
            wins += 1

    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    # One-sided p: how often resampling failed to reproduce the improvement.
    p_value = 1.0 - wins / settings.bootstrap_resamples
    return {
        "metric": metric,
        "baseline_score": round(base_score, 4),
        "system_score": round(sys_score, 4),
        "delta": round(observed, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "p_value": round(p_value, 4),
        "resamples": settings.bootstrap_resamples,
        "n_docs": n,
    }

"""The DAG-context translation condition's context strategy.

A segment's context is its parents in the discourse graph
(`DiscourseGraph.ancestors`), rendered as source/translation pairs so that
consistency ties back to how earlier segments were actually rendered, not
just what they said. Nearest ancestors are kept when the token budget is
tight, because the nearest context is the most likely to carry the pronoun or
term the current segment depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..io.text import estimate_tokens
from ..schemas import Segment, TranslationRecord


@dataclass
class ContextBlock:
    """What a segment is shown before its own text."""

    text: str
    seg_ids: list[str] = field(default_factory=list)
    truncated: bool = False


def _pair_lines(
    seg_ids: list[str],
    segments_by_id: dict[str, Segment],
    records: dict[str, TranslationRecord],
) -> list[str]:
    """Render ancestor segments as source/translation pairs.

    The translation is included when available, because consistency is the
    thing context is supposed to buy: showing only the source tells the model
    what was said, not how it was rendered.
    """
    lines: list[str] = []
    for sid in seg_ids:
        seg = segments_by_id[sid]
        lines.append(f"[{seg.order}] source: {seg.text}")
        rec = records.get(sid)
        if rec is not None:
            lines.append(f"[{seg.order}] translation: {rec.target_text}")
    return lines


def build_dag_context(
    parent_ids: list[str],
    segments_by_id: dict[str, Segment],
    records: dict[str, TranslationRecord],
    settings: Settings,
) -> ContextBlock:
    """The context block for a segment: its discourse-graph ancestors.

    `parent_ids` is expected in ascending reading order (as
    `DiscourseGraph.parents`/`ancestors` return it), so dropping from the
    front drops the furthest-back ancestor first until the block fits the
    token budget.
    """
    if not parent_ids:
        return ContextBlock(text="", seg_ids=[])

    budget = settings.translation.context_token_budget
    cpt = settings.translation.chars_per_token_estimate
    kept = list(parent_ids)
    truncated = False
    while kept:
        text = "\n".join(_pair_lines(kept, segments_by_id, records))
        if estimate_tokens(text, cpt) <= budget:
            return ContextBlock(text=text, seg_ids=kept, truncated=truncated)
        kept.pop(0)
        truncated = True
    return ContextBlock(text="", seg_ids=[], truncated=truncated)

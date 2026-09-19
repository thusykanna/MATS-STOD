"""GRAFT's edge agent.

Faithful to Dutta et al. (2025): every segment is linked to its successor, and
every non-adjacent ordered pair is judged by one yes/no call. Cost is
quadratic in segments per document — exactly (n-1)(n-2)/2 pairwise calls — and
that cost is the reason for every guard in this module (D2).

Three properties are deliberate:

1. The number of calls is computed and checked *before* the first one is made.
   A ceiling discovered halfway through a document has already been paid for.
2. An unparseable answer counts as "not connected" (D18), the conservative
   reading: a missing edge weakens the context, an invented one corrupts it.
3. The model answers yes or no and nothing else, so it can neither name a
   relation nor return text. Every edge it produces carries the untyped label
   `graft_dependency` with no evidence span (D6).
"""

from __future__ import annotations

from ..config import Settings
from ..llm.base import Message, ParseError, parse_yes_no
from ..llm.client import CachedLLM
from ..prompts.registry import render
from ..schemas import Edge, Segment
from .edge_inference import EdgeInferenceResult, EdgeInferrer


class PairwiseCallBudgetError(RuntimeError):
    """Raised when a document would need more pairwise calls than allowed."""


def plan_pairs(n: int, max_distance: int | None = None) -> list[tuple[int, int]]:
    """Non-adjacent ordered pairs (i, j) with j >= i + 2, in reading order.

    Adjacent pairs are excluded because GRAFT links every segment to its
    successor unconditionally; asking about them would pay for an answer that
    is already decided.
    """
    pairs = [(i, j) for j in range(n) for i in range(j - 1)]
    if max_distance is not None:
        pairs = [(i, j) for i, j in pairs if j - i <= max_distance]
    return pairs


class GraftPairwiseEdgeInferrer(EdgeInferrer):
    name = "graft"

    def __init__(self, settings: Settings, llm: CachedLLM) -> None:
        self.settings = settings
        self.graph_settings = settings.graph
        self.llm = llm

    def infer(self, segments: list[Segment]) -> EdgeInferenceResult:
        ordered = sorted(segments, key=lambda s: s.order)
        n = len(ordered)
        if n < 2:
            return EdgeInferenceResult(
                edges=[], stats={"inferrer": self.name, "llm_calls": 0, "n_planned_calls": 0}
            )

        pairs = plan_pairs(n, self.graph_settings.max_pair_distance)
        self._check_budget(n, len(pairs))

        # Predecessor edges are unconditional in GRAFT and cost nothing, so
        # they are built first: if the budget check above had raised, no call
        # would have been made and no money spent.
        edges: list[Edge] = [
            Edge(
                src=ordered[i].seg_id,
                dst=ordered[i + 1].seg_id,
                type="continuation",
                origin="structural",
                evidence=None,
            )
            for i in range(n - 1)
        ]

        calls = cache_hits = unparsed = yes_count = 0
        for i, j in pairs:
            decision, was_cached, ok = self._ask(ordered[i].text, ordered[j].text)
            calls += 1
            cache_hits += int(was_cached)
            if not ok:
                unparsed += 1
            if decision:
                yes_count += 1
                edges.append(
                    Edge(
                        src=ordered[i].seg_id,
                        dst=ordered[j].seg_id,
                        type="graft_dependency",
                        origin="llm",
                        evidence=None,
                        confidence=None,
                    )
                )

        return EdgeInferenceResult(
            edges=edges,
            stats={
                "inferrer": self.name,
                "n_planned_calls": len(pairs),
                "llm_calls": calls,
                "llm_cache_hits": cache_hits,
                "decisions_yes": yes_count,
                "decisions_unparsed": unparsed,
                # The share of pairs answered "yes" decides how dense the graph
                # is, and a dense graph collapses D1 toward B0. It is reported
                # on every run so that finding arrives before M4 is designed.
                "yes_rate": round(yes_count / len(pairs), 4) if pairs else 0.0,
                "n_predecessor_edges": n - 1,
                "prompt_version": self.graph_settings.prompt_version,
            },
        )

    # -- helpers ----------------------------------------------------------

    def _check_budget(self, n_segments: int, n_pairs: int) -> None:
        """Refuse a document that would exceed the per-document call ceiling.

        Checked before the first call rather than during the loop: the point
        of a budget is to prevent the spend, and a run aborted halfway has
        already paid for what it did.
        """
        ceiling = self.graph_settings.max_pairwise_calls_per_doc
        if ceiling is not None and n_pairs > ceiling:
            raise PairwiseCallBudgetError(
                f"this document has {n_segments} segments, which needs {n_pairs} pairwise "
                f"calls, above the ceiling of {ceiling} "
                f"(graph.max_pairwise_calls_per_doc).\n\n"
                "What to do:\n"
                "  Raise the ceiling if you mean to spend it, or set\n"
                "  graph.max_pair_distance to bound how far apart a judged pair\n"
                "  may be. Note that bounding distance is a departure from GRAFT\n"
                "  and must be reported as one."
            )

    def _ask(self, first: str, second: str) -> tuple[bool, bool, bool]:
        """Ask whether the second discourse depends on the first.

        Returns (decision, cached, parsed_ok).
        """
        prompt = render(
            self.graph_settings.prompt_version,
            source_lang=self.settings.langs.source_name,
            target_lang=self.settings.langs.target_name,
            discourse_1=first,
            discourse_2=second,
        )
        response = self.llm.complete(
            [Message("user", prompt)],
            purpose="edge_inference",
            max_output_tokens=self.graph_settings.decision_max_tokens,
            thinking_budget=self.graph_settings.decision_thinking_budget,
            temperature=0.0,
        )
        try:
            return parse_yes_no(response.text), response.cached, True
        except ParseError:
            return False, response.cached, False

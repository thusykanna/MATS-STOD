"""Edge inferrer interface, the `none` control, and the registry.

Which edges a discourse graph should carry is the open question this project
exists to answer, so every candidate answer lives behind one interface and is
chosen by config, never by an import. The same arrangement is already used for
segmenters.

The registry below names every inferrer the config accepts, including ones
that are not written yet. A missing implementation therefore fails with a
message naming the file to create, rather than with an import traceback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Edge, Segment

#: Inferrer names accepted by `graph.edge_inferrer`, with the module that owns
#: each one. Keeping this in one place is what lets two people implement
#: different inferrers without editing the same file.
INFERRER_MODULES: dict[str, str] = {
    "none": "graph.edge_inference",
    "predecessor": "graph.simple_edges",
    "tfidf": "graph.simple_edges",
    "graft": "graph.graft_edges",
}


@dataclass
class EdgeInferenceResult:
    """Edges plus everything worth reporting about how they were produced.

    Mirrors `SegmentationResult`: the stats travel with the output because a
    run that cost 1,176 LLM calls and a run that cost none are not comparable
    on score alone.
    """

    edges: list[Edge]
    stats: dict[str, Any] = field(default_factory=dict)


class EdgeInferrer(ABC):
    """Proposes dependency edges over one document's segments.

    An inferrer proposes; it does not decide. Deduplication, direction, the
    parent cap and validation all happen in `assemble.py`, so an inferrer may
    emit duplicates or backward pairs without corrupting the graph.
    """

    name: str = "base"

    @abstractmethod
    def infer(self, segments: list[Segment]) -> EdgeInferenceResult:
        """Propose edges over `segments`, which are in reading order."""


class NoneEdgeInferrer(EdgeInferrer):
    """Proposes nothing.

    The control condition. A graph with no edges makes D1 degenerate to B1,
    which is the floor every other inferrer has to beat.
    """

    name = "none"

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings

    def infer(self, segments: list[Segment]) -> EdgeInferenceResult:
        return EdgeInferenceResult(edges=[], stats={"inferrer": self.name, "llm_calls": 0})


def build_edge_inferrer(name: str, settings, llm=None):  # noqa: ANN001 - avoids a config import cycle
    """Instantiate an edge inferrer by config name.

    An inferrer that is named in the config but not yet written raises a
    message naming the module to create. This is deliberate: the registry is
    complete from the start so that adding an inferrer never means editing
    this function, and therefore never means a merge conflict with whoever is
    writing a different one.
    """
    if name == "none":
        return NoneEdgeInferrer(settings)

    if name in {"predecessor", "tfidf"}:
        try:
            from . import simple_edges
        except ImportError as exc:  # pragma: no cover - exercised by the message test
            raise NotImplementedError(_not_built(name)) from exc
        cls = {
            "predecessor": "PredecessorEdgeInferrer",
            "tfidf": "TfidfEdgeInferrer",
        }[name]
        if not hasattr(simple_edges, cls):
            raise NotImplementedError(_not_built(name))
        return getattr(simple_edges, cls)(settings)

    if name == "graft":
        try:
            from .graft_edges import GraftPairwiseEdgeInferrer
        except ImportError as exc:
            raise NotImplementedError(_not_built(name)) from exc
        if llm is None:
            raise ValueError("the graft edge inferrer needs an LLM client")
        return GraftPairwiseEdgeInferrer(settings, llm)

    raise ValueError(f"unknown edge inferrer: {name!r}. Available: {sorted(INFERRER_MODULES)}")


def _not_built(name: str) -> str:
    module = INFERRER_MODULES[name]
    return (
        f"the {name!r} edge inferrer is not implemented yet. "
        f"It belongs in src/mats_stod/{module.replace('.', '/')}.py. "
        f"Until it lands, run with --edges none."
    )

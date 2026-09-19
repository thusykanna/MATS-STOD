"""Edge inferrer interface.

Kept as an interface, not a registry: GRAFT is the only inferrer for now, but
a future comparison method implements this ABC rather than being wired in ad
hoc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Edge, Segment


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

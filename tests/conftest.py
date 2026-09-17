"""Shared fixtures, and the guarantee that no test opens a socket."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from mats_stod.config import Settings, load_settings
from mats_stod.io.text import read_text
from mats_stod.parsing.plaintext import PlainTextParser
from mats_stod.schemas import Document

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach the network.

    The build constraint is that the suite runs offline. Enforcing it here
    means a provider import that quietly phones home is caught by the test
    that triggers it, not by a surprising bill.
    """

    def blocked(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("network access is not allowed in tests")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default settings, redirected so tests never touch the real run dirs."""
    s = load_settings()
    s.paths.runs = str(tmp_path / "runs")
    s.paths.gold_segmentation = str(tmp_path / "gold" / "segmentation")
    s.paths.gold_edges = str(tmp_path / "gold" / "edges")
    s.paths.split_file = str(tmp_path / "splits.json")
    s.llm.cache_path = str(tmp_path / "cache.sqlite")
    return s


@pytest.fixture
def sample_text() -> str:
    return read_text(SAMPLES / "circular_01" / "circular_01.si")


@pytest.fixture
def sample_doc(sample_text: str) -> Document:
    return PlainTextParser().parse(sample_text, "circular_01", "si")

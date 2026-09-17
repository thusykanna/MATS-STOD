"""Loading credentials from .env, without leaking them."""

from __future__ import annotations

import os

import pytest

from mats_stod.io.env import ALLOWED, credential_status, load_env_file


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ALLOWED:
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path, body: str):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nothing.env") == []


def test_values_are_loaded(tmp_path):
    p = write_env(tmp_path, "GOOGLE_CLOUD_PROJECT=my-project\nGOOGLE_CLOUD_LOCATION=us-central1\n")
    applied = load_env_file(p)
    assert set(applied) == {"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"}
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "my-project"


def test_comments_blank_lines_and_export_prefix(tmp_path):
    p = write_env(tmp_path, "# a comment\n\nexport GEMINI_API_KEY=abc123\n")
    load_env_file(p)
    assert os.environ["GEMINI_API_KEY"] == "abc123"


def test_quotes_are_stripped(tmp_path):
    p = write_env(tmp_path, 'GOOGLE_CLOUD_PROJECT="quoted-project"\n')
    load_env_file(p)
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "quoted-project"


def test_the_real_environment_wins(tmp_path, monkeypatch):
    """An explicit export must still override the file."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-shell")
    p = write_env(tmp_path, "GOOGLE_CLOUD_PROJECT=from-file\n")
    assert load_env_file(p) == []
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "from-shell"


def test_unlisted_variables_are_ignored(tmp_path):
    """A stray line must not be able to change PATH or anything else."""
    before = os.environ.get("PATH")
    p = write_env(tmp_path, "PATH=/tmp/evil\nPYTHONPATH=/tmp/evil\n")
    assert load_env_file(p) == []
    assert os.environ.get("PATH") == before


def test_credential_status_never_prints_a_secret(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    status = credential_status()
    assert "super-secret-value" not in str(status)
    assert status["GEMINI_API_KEY"] == "set (18 chars)"


def test_credential_status_shows_non_secret_values(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    assert credential_status()["GOOGLE_CLOUD_PROJECT"] == "my-project"
    assert credential_status()["GEMINI_API_KEY"] == "not set"

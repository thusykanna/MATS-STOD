"""Classifying and explaining Vertex errors.

These are pure functions over the error text, so they run offline and need no
Google SDK. They matter because a setup mistake that looks like a crash costs
a new user far more time than one that names its own fix.
"""

from __future__ import annotations

import pytest

from mats_stod.llm.gemini import explain_error, is_permanent_error

RETIRED_MODEL = (
    "404 NOT_FOUND. Publisher model projects/p/locations/us-central1/publishers/"
    "google/models/gemini-2.0-flash-001 was not found or your project does not "
    "have access to it."
)
NO_PROJECT = "403 PERMISSION_DENIED. Permission denied on resource project bad-id."
API_DISABLED = "403 SERVICE_DISABLED. Vertex AI API has not been used in project 1 before."
NO_ADC = "DefaultCredentialsError: could not automatically determine credentials"
QUOTA = "429 RESOURCE_EXHAUSTED. Quota exceeded for requests."
BLIP = "503 UNAVAILABLE. The service is currently unavailable."


@pytest.mark.parametrize("text", [RETIRED_MODEL, NO_PROJECT, API_DISABLED])
def test_permanent_errors_are_not_retried(text):
    """Retrying these wastes the user's time and changes nothing."""
    assert is_permanent_error(Exception(text)) is True


@pytest.mark.parametrize("text", [QUOTA, BLIP])
def test_transient_errors_are_still_retried(text):
    """Quota and server blips do clear, so the backoff must still apply."""
    assert is_permanent_error(Exception(text)) is False


def test_retired_model_hint_points_at_the_model_setting():
    out = explain_error(Exception(RETIRED_MODEL))
    assert "llm.model" in out
    assert "gemini-2.5-flash" in out


def test_disabled_api_hint_gives_the_enable_command():
    assert "gcloud services enable aiplatform.googleapis.com" in explain_error(
        Exception(API_DISABLED)
    )


def test_permission_hint_mentions_billing_and_the_login_command():
    out = explain_error(Exception(NO_PROJECT))
    assert "billing" in out
    assert "gcloud auth application-default login" in out


def test_missing_credentials_hint_gives_the_login_command():
    assert "gcloud auth application-default login" in explain_error(Exception(NO_ADC))


def test_the_original_error_is_never_discarded():
    """The hint is added to the provider's text, not substituted for it."""
    out = explain_error(Exception(RETIRED_MODEL))
    assert RETIRED_MODEL in out


def test_an_unrecognised_error_passes_through_unchanged():
    assert explain_error(Exception(BLIP)) == BLIP


# -- truncation guard -----------------------------------------------------


class _FakeCandidate:
    def __init__(self, reason: str) -> None:
        self.finish_reason = reason


class _FakeResponse:
    def __init__(self, reason: str) -> None:
        self.candidates = [_FakeCandidate(reason)]


def test_max_tokens_finish_is_detected():
    from mats_stod.llm.gemini import _hit_token_cap

    assert _hit_token_cap(_FakeResponse("FinishReason.MAX_TOKENS")) is True


def test_normal_finish_is_not_treated_as_truncation():
    from mats_stod.llm.gemini import _hit_token_cap

    assert _hit_token_cap(_FakeResponse("FinishReason.STOP")) is False


def test_a_malformed_response_does_not_crash_the_check():
    from mats_stod.llm.gemini import _hit_token_cap

    assert _hit_token_cap(object()) is False
    assert _hit_token_cap(None) is False

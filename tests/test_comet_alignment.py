"""COMET alignment checks without loading a model or accessing the network."""

import pytest

from mats_stod.evaluation.comet_metric import CometMetric, score_documents


def test_comet_rejects_misaligned_documents_before_loading_model():
    with pytest.raises(ValueError, match="zip"):
        CometMetric().score(["source"], ["hypothesis"], [])
    with pytest.raises(ValueError, match="zip"):
        score_documents(CometMetric(), ["source"], [], ["reference"])


@pytest.mark.parametrize("scores", [[], [0.5, 0.6]])
def test_comet_rejects_wrong_number_of_scores(scores):
    class FakeMetric:
        def score(self, sources, hypotheses, references):
            return scores

    with pytest.raises(ValueError, match="zip"):
        score_documents(FakeMetric(), ["source"], ["hypothesis"], ["reference"])

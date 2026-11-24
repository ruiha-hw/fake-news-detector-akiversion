# test_runtime_interpretation_policy.py
# Verifies the validation-audited 0.75 runtime interpretation policy.

from __future__ import annotations

import pytest

from model.model_service import (
    INTERPRETATION_THRESHOLD,
    ModelIntegrityError,
    PredictionResult,
    interpret_prediction,
)


def test_interpretation_threshold_is_locked_at_075() -> None:
    assert INTERPRETATION_THRESHOLD == 0.75


@pytest.mark.parametrize("label", [0, 1])
def test_below_threshold_is_uncertain_for_either_raw_label(label: int) -> None:
    decisive, classification, guidance = interpret_prediction(label, 0.749999)
    assert decisive is False
    assert classification == "seek further information"
    assert "Seek confirmation" in guidance


def test_threshold_is_inclusive_for_fake_pattern_result() -> None:
    decisive, classification, guidance = interpret_prediction(0, 0.75)
    assert decisive is True
    assert classification == "potentially misleading"
    assert "fake-labelled" in guidance


def test_threshold_is_inclusive_for_real_pattern_result() -> None:
    decisive, classification, guidance = interpret_prediction(1, 0.75)
    assert decisive is True
    assert classification == "more consistent with real-news patterns"
    assert "does not confirm" in guidance


def test_interpretation_does_not_change_raw_label() -> None:
    raw_label = 1
    decisive, classification, guidance = interpret_prediction(raw_label, 0.66)
    result = PredictionResult(
        label=raw_label,
        classification=classification,
        confidence=0.66,
        fake_probability=0.34,
        real_probability=0.66,
        input_characters=100,
        decisive=decisive,
        guidance=guidance,
        model_classification="likely real",
    )
    payload = result.to_dict()
    assert payload["label"] == 1
    assert payload["model_classification"] == "likely real"
    assert payload["classification"] == "seek further information"
    assert payload["decisive"] is False


@pytest.mark.parametrize("threshold", [0.49, 1.01])
def test_invalid_policy_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ModelIntegrityError, match="between 0.5 and 1.0"):
        interpret_prediction(0, 0.90, threshold)


def test_unsupported_label_is_rejected() -> None:
    with pytest.raises(ModelIntegrityError, match="Unsupported model label"):
        interpret_prediction(2, 0.90)

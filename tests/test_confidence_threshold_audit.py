# test_confidence_threshold_audit.py
# Verifies confidence buckets, abstention coverage, class summaries, and calibration output.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.confidence_threshold_audit import (
    ThresholdAuditError,
    audit_calibration,
    audit_confidence_buckets,
    audit_threshold_policy,
    build_baseline_metrics,
    build_validation_predictions,
    select_policy_examples,
)
from model.dataset_preparation import LABEL_COLUMN, SPLIT_COLUMN
from model.model_experimentation import ExperimentConfiguration, build_pipeline
from model.preprocessing import TEXT_COLUMN, TITLE_COLUMN


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual_label": [0, 0, 1, 1, 0, 1],
            "predicted_label": [0, 1, 1, 1, 0, 0],
            "fake_probability": [0.95, 0.40, 0.20, 0.10, 0.70, 0.45],
            "real_probability": [0.05, 0.60, 0.80, 0.90, 0.30, 0.55],
            "confidence": [0.95, 0.60, 0.80, 0.90, 0.70, 0.55],
            "correct": [True, False, True, True, True, False],
        }
    )


def fitted_pipeline():
    texts = [
        "TITLE:\nfake one\n\nARTICLE:\nhoax fabricated",
        "TITLE:\nfake two\n\nARTICLE:\nfalse invented",
        "TITLE:\nreal one\n\nARTICLE:\nofficial confirmed",
        "TITLE:\nreal two\n\nARTICLE:\nverified evidence",
    ]
    labels = [0, 0, 1, 1]
    configuration = ExperimentConfiguration(
        max_features=100,
        min_df=1,
        max_df=1.0,
        max_iterations=200,
    )
    pipeline = build_pipeline(configuration)
    pipeline.fit(texts, labels)
    return pipeline


def validation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TITLE_COLUMN: ["fake one", "real one", "fake two", "real two"],
            TEXT_COLUMN: [
                "hoax fabricated",
                "official confirmed",
                "false invented",
                "verified evidence",
            ],
            LABEL_COLUMN: [0, 1, 0, 1],
            SPLIT_COLUMN: ["validation"] * 4,
        }
    )


def test_build_validation_predictions_returns_probabilities_and_correctness() -> None:
    result = build_validation_predictions(validation_data(), fitted_pipeline(), "combined")
    assert len(result) == 4
    assert set(result.columns) == {
        "actual_label",
        "predicted_label",
        "fake_probability",
        "real_probability",
        "confidence",
        "correct",
    }
    assert np.allclose(
        result["fake_probability"] + result["real_probability"],
        1.0,
    )


def test_build_validation_predictions_rejects_empty_validation_split() -> None:
    data = validation_data()
    data[SPLIT_COLUMN] = "train"
    with pytest.raises(ThresholdAuditError, match="contains no records"):
        build_validation_predictions(data, fitted_pipeline(), "combined")


def test_confidence_buckets_cover_every_record_once() -> None:
    rows = audit_confidence_buckets(sample_predictions())
    assert sum(int(row["records"]) for row in rows) == 6
    assert sum(float(row["share"] or 0.0) for row in rows) == pytest.approx(1.0)


def test_threshold_policy_calculates_coverage_and_selective_accuracy() -> None:
    rows = audit_threshold_policy(sample_predictions(), thresholds=[0.75])
    row = rows[0]
    assert row["decisive_records"] == 3
    assert row["uncertain_records"] == 3
    assert row["coverage"] == 0.5
    assert row["selective_accuracy"] == 1.0


def test_threshold_policy_reports_fake_and_real_decisions_separately() -> None:
    row = audit_threshold_policy(sample_predictions(), thresholds=[0.50])[0]
    assert row["fake_decisions"]["records"] == 3
    assert row["real_decisions"]["records"] == 3
    assert row["fake_decisions"]["precision_within_decisions"] == pytest.approx(2 / 3)
    assert row["real_decisions"]["precision_within_decisions"] == pytest.approx(2 / 3)


def test_higher_threshold_does_not_increase_coverage() -> None:
    rows = audit_threshold_policy(sample_predictions(), thresholds=[0.50, 0.75, 0.90])
    coverages = [float(row["coverage"]) for row in rows]
    assert coverages == sorted(coverages, reverse=True)


def test_calibration_audit_returns_brier_log_loss_and_bins() -> None:
    result = audit_calibration(sample_predictions())
    assert result["brier_score"] >= 0.0
    assert result["log_loss"] >= 0.0
    assert result["expected_calibration_error_top_class"] >= 0.0
    assert result["bins_for_real_label_probability"]


def test_baseline_metrics_match_default_predictions() -> None:
    result = build_baseline_metrics(sample_predictions())
    assert result["records"] == 6
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert 0.5 <= result["minimum_confidence"] <= 1.0


def test_policy_review_does_not_select_threshold() -> None:
    rows = audit_threshold_policy(sample_predictions())
    review = select_policy_examples(rows)
    assert review["selection_status"] == "not_selected"
    assert review["candidate_rows"]

# test_final_model_evaluation.py
# Verifies final-stage mode locking, test evaluation, refit isolation, and guards.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.dataset_preparation import LABEL_COLUMN, SPLIT_COLUMN
from model.final_model_evaluation import (
    FinalEvaluationError,
    evaluate_test_split,
    get_selected_configuration,
    get_selected_mode,
    refit_deployable_pipeline,
)
from model.model_experimentation import ExperimentConfiguration, build_pipeline
from model.preprocessing import TEXT_COLUMN, TITLE_COLUMN


def make_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for index in range(20):
        split = "train" if index < 14 else "validation" if index < 17 else "test"
        rows.append(
            {
                TITLE_COLUMN: f"fabricated rumour {index}",
                TEXT_COLUMN: f"invented hoax fabricated claim {index}",
                LABEL_COLUMN: 0,
                SPLIT_COLUMN: split,
            }
        )
        rows.append(
            {
                TITLE_COLUMN: f"verified report {index}",
                TEXT_COLUMN: f"official confirmed factual evidence {index}",
                LABEL_COLUMN: 1,
                SPLIT_COLUMN: split,
            }
        )

    return pd.DataFrame(rows)


def fitted_training_pipeline(data: pd.DataFrame):
    train = data.loc[data[SPLIT_COLUMN].eq("train")]
    configuration = ExperimentConfiguration(
        max_features=500,
        min_df=1,
        max_df=1.0,
        max_iterations=200,
    )
    pipeline = build_pipeline(configuration)
    features = (
        "TITLE:\n"
        + train[TITLE_COLUMN]
        + "\n\nARTICLE:\n"
        + train[TEXT_COLUMN]
    )
    pipeline.fit(features, train[LABEL_COLUMN])
    return pipeline, configuration


def test_get_selected_mode_accepts_combined() -> None:
    assert get_selected_mode({"selected_feature_mode": "combined"}) == "combined"


def test_get_selected_mode_rejects_unknown_value() -> None:
    with pytest.raises(FinalEvaluationError, match="Unsupported selected"):
        get_selected_mode({"selected_feature_mode": "unknown"})


def test_get_selected_configuration_reconstructs_values() -> None:
    expected = ExperimentConfiguration(max_features=1234, classifier_c=0.5)
    actual = get_selected_configuration({"configuration": expected.__dict__})
    assert actual == expected


def test_get_selected_configuration_rejects_missing_configuration() -> None:
    with pytest.raises(FinalEvaluationError, match="missing or malformed"):
        get_selected_configuration({})


def test_evaluate_test_split_uses_test_records_only() -> None:
    data = make_data()
    pipeline, _ = fitted_training_pipeline(data)
    result = evaluate_test_split(data, pipeline, "combined")
    assert result["test_rows"] == 6
    assert result["probability_shape"] == [6, 2]
    assert result["classes"] == [0, 1]
    assert result["metrics"]["accuracy"] == 1.0


def test_evaluate_test_split_rejects_missing_test_records() -> None:
    data = make_data()
    data = data.loc[data[SPLIT_COLUMN].ne("test")]
    pipeline, _ = fitted_training_pipeline(data)
    with pytest.raises(FinalEvaluationError, match="contains no records"):
        evaluate_test_split(data, pipeline, "combined")


def test_refit_uses_train_and_validation_but_not_test() -> None:
    data = make_data()
    _, configuration = fitted_training_pipeline(data)
    _, result = refit_deployable_pipeline(data, "combined", configuration)
    assert result["refit_rows"] == 34
    assert result["refit_class_counts"] == {"0": 17, "1": 17}


def test_refit_pipeline_has_predict_proba_and_binary_classes() -> None:
    data = make_data()
    _, configuration = fitted_training_pipeline(data)
    pipeline, _ = refit_deployable_pipeline(data, "combined", configuration)
    assert hasattr(pipeline, "predict_proba")
    assert np.array_equal(pipeline.classes_, np.array([0, 1]))

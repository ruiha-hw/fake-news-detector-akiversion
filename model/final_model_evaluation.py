# final_model_evaluation.py
# Performs one held-out test evaluation, then refits and saves the deployable pipeline.

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from model.dataset_preparation import (
    LABEL_COLUMN,
    SPLIT_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    calculate_file_hash,
)
from model.model_experimentation import (
    ExperimentConfiguration,
    ExperimentError,
    RESULTS_PATH,
    SELECTED_PIPELINE_PATH,
    build_pipeline,
    evaluate_predictions,
    load_experiment_data,
    make_feature_series,
)
from model.preprocessing import FeatureMode


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
FINAL_ARTIFACT_DIRECTORY: Final[Path] = PROJECT_ROOT / "model_artifacts" / "final"
FINAL_PIPELINE_PATH: Final[Path] = FINAL_ARTIFACT_DIRECTORY / "fake_news_pipeline.joblib"
FINAL_METRICS_PATH: Final[Path] = FINAL_ARTIFACT_DIRECTORY / "final_test_metrics.json"
FINAL_METADATA_PATH: Final[Path] = FINAL_ARTIFACT_DIRECTORY / "model_metadata.json"

EXPECTED_LABELS: Final[list[int]] = [0, 1]


class FinalEvaluationError(ValueError):
    """Raised when the locked final-evaluation contract cannot be satisfied."""


def load_locked_experiment() -> tuple[dict[str, object], object]:
    """Loads and verifies the validation-stage report and selected pipeline."""

    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Validation report not found: {RESULTS_PATH}")

    if not SELECTED_PIPELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Selected validation pipeline not found: {SELECTED_PIPELINE_PATH}"
        )

    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    if report.get("status") != "validation_only":
        raise FinalEvaluationError("Validation report status is not validation_only.")

    if report.get("test_split_evaluated") is not False:
        raise FinalEvaluationError("Validation report indicates prior test evaluation.")

    expected_hash = str(report.get("selected_pipeline_sha256", ""))
    actual_hash = calculate_file_hash(SELECTED_PIPELINE_PATH, "sha256")

    if actual_hash != expected_hash:
        raise FinalEvaluationError(
            "Selected pipeline SHA-256 mismatch. "
            f"Expected {expected_hash}, found {actual_hash}."
        )

    pipeline = joblib.load(SELECTED_PIPELINE_PATH)

    if list(getattr(pipeline, "named_steps", {}).keys()) != [
        "tfidf",
        "classifier",
    ]:
        raise FinalEvaluationError("Selected artefact is not the expected pipeline.")

    return report, pipeline


def get_selected_configuration(report: dict[str, object]) -> ExperimentConfiguration:
    """Reconstructs the exact configuration selected during validation."""

    raw = report.get("configuration")

    if not isinstance(raw, dict):
        raise FinalEvaluationError("Validation configuration is missing or malformed.")

    return ExperimentConfiguration(**raw)


def get_selected_mode(report: dict[str, object]) -> FeatureMode:
    """Returns the locked feature mode selected on validation macro F1."""

    mode = report.get("selected_feature_mode")

    if mode not in {"title", "text", "combined"}:
        raise FinalEvaluationError(f"Unsupported selected feature mode: {mode!r}")

    return mode


def evaluate_test_split(
    data: pd.DataFrame,
    pipeline: object,
    mode: FeatureMode,
) -> dict[str, object]:
    """Evaluates the selected training-only pipeline once on held-out test data."""

    test = data.loc[data[SPLIT_COLUMN].eq(TEST_SPLIT)].copy()

    if test.empty:
        raise FinalEvaluationError("Held-out test split contains no records.")

    features = make_feature_series(test, mode)
    started = time.perf_counter()
    predictions = pipeline.predict(features)
    prediction_seconds = time.perf_counter() - started
    metrics = evaluate_predictions(test[LABEL_COLUMN], predictions)

    probabilities = pipeline.predict_proba(features)

    return {
        "feature_mode": mode,
        "test_rows": int(len(test)),
        "prediction_seconds": round(float(prediction_seconds), 4),
        "mean_prediction_milliseconds": round(
            float(prediction_seconds * 1000.0 / len(test)),
            6,
        ),
        "metrics": metrics,
        "probability_shape": [int(value) for value in probabilities.shape],
        "classes": [int(value) for value in pipeline.classes_.tolist()],
    }


def refit_deployable_pipeline(
    data: pd.DataFrame,
    mode: FeatureMode,
    configuration: ExperimentConfiguration,
) -> tuple[object, dict[str, object]]:
    """Refits the locked configuration on training plus validation records only."""

    development = data.loc[
        data[SPLIT_COLUMN].isin([TRAIN_SPLIT, VALIDATION_SPLIT])
    ].copy()

    if development.empty:
        raise FinalEvaluationError("Training and validation records are unavailable.")

    features = make_feature_series(development, mode)
    pipeline = build_pipeline(configuration)
    started = time.perf_counter()
    pipeline.fit(features, development[LABEL_COLUMN])
    fit_seconds = time.perf_counter() - started

    if [int(value) for value in pipeline.classes_.tolist()] != EXPECTED_LABELS:
        raise FinalEvaluationError("Final pipeline classes are not [0, 1].")

    return pipeline, {
        "refit_rows": int(len(development)),
        "refit_class_counts": {
            str(key): int(value)
            for key, value in development[LABEL_COLUMN]
            .value_counts()
            .sort_index()
            .items()
        },
        "fit_seconds": round(float(fit_seconds), 4),
        "feature_count": int(
            len(pipeline.named_steps["tfidf"].get_feature_names_out())
        ),
        "classifier_iterations": [
            int(value)
            for value in pipeline.named_steps["classifier"].n_iter_.tolist()
        ],
    }


def run_final_stage() -> tuple[dict[str, object], dict[str, object]]:
    """Executes the one-time test evaluation and final development-set refit."""

    if FINAL_METRICS_PATH.exists() or FINAL_PIPELINE_PATH.exists():
        raise FinalEvaluationError(
            "Final outputs already exist. Refusing to repeat held-out test evaluation."
        )

    validation_report, selected_pipeline = load_locked_experiment()
    selected_mode = get_selected_mode(validation_report)
    configuration = get_selected_configuration(validation_report)
    data, preparation_metadata = load_experiment_data()

    test_result = evaluate_test_split(data, selected_pipeline, selected_mode)
    final_pipeline, refit_result = refit_deployable_pipeline(
        data,
        selected_mode,
        configuration,
    )

    FINAL_ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, FINAL_PIPELINE_PATH, compress=3)
    final_hash = calculate_file_hash(FINAL_PIPELINE_PATH, "sha256")

    metrics_report: dict[str, object] = {
        "status": "final_test_complete",
        "test_split_evaluated": True,
        "test_evaluation_count": 1,
        "selected_feature_mode": selected_mode,
        "selection_metric": validation_report["selection_metric"],
        "validation_candidates": validation_report["candidates"],
        "held_out_test": test_result,
    }

    metadata: dict[str, object] = {
        "model_name": "WELFake TF-IDF Logistic Regression",
        "model_role": "educational fake-news text classifier",
        "label_mapping": {
            "0": "likely fake",
            "1": "likely real",
        },
        "feature_mode": selected_mode,
        "configuration": asdict(configuration),
        "dataset_preparation": preparation_metadata,
        "final_refit": refit_result,
        "pipeline_path": str(FINAL_PIPELINE_PATH),
        "pipeline_sha256": final_hash,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "The output is a dataset-trained classification, not factual verification.",
            "Confidence is model probability, not proof that a claim is true or false.",
            "The model should be loaded only in the recorded compatible environment.",
        ],
    }

    FINAL_METRICS_PATH.write_text(
        json.dumps(metrics_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    FINAL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return metrics_report, metadata


def main() -> None:
    print("Running the single locked held-out test evaluation...")
    metrics, metadata = run_final_stage()
    print(json.dumps(metrics["held_out_test"], indent=2, ensure_ascii=False))
    print(f"Final pipeline: {FINAL_PIPELINE_PATH}")
    print(f"Final metrics: {FINAL_METRICS_PATH}")
    print(f"Model metadata: {FINAL_METADATA_PATH}")
    print(f"Final pipeline SHA-256: {metadata['pipeline_sha256']}")


if __name__ == "__main__":
    main()

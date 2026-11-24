# confidence_threshold_audit.py
# Audits validation probabilities, calibration, and abstention thresholds without changing the model.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
)

from model.dataset_preparation import (
    LABEL_COLUMN,
    SPLIT_COLUMN,
    VALIDATION_SPLIT,
    calculate_file_hash,
)
from model.model_experimentation import (
    RESULTS_PATH,
    SELECTED_PIPELINE_PATH,
    ExperimentError,
    load_experiment_data,
    make_feature_series,
)
from model.preprocessing import FeatureMode


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
AUDIT_DIRECTORY: Final[Path] = PROJECT_ROOT / "model_artifacts" / "threshold_audit"
AUDIT_JSON_PATH: Final[Path] = AUDIT_DIRECTORY / "confidence_threshold_audit.json"
THRESHOLD_CSV_PATH: Final[Path] = AUDIT_DIRECTORY / "threshold_policy.csv"
CONFIDENCE_BUCKET_CSV_PATH: Final[Path] = AUDIT_DIRECTORY / "confidence_buckets.csv"
CALIBRATION_CSV_PATH: Final[Path] = AUDIT_DIRECTORY / "calibration_bins.csv"

THRESHOLDS: Final[tuple[float, ...]] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
CONFIDENCE_EDGES: Final[tuple[float, ...]] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.0000001,
)
CALIBRATION_BINS: Final[int] = 10


class ThresholdAuditError(ValueError):
    """Raised when the locked validation audit contract is not satisfied."""


def load_locked_validation_pipeline() -> tuple[dict[str, object], object]:
    """Verifies and loads the training-only pipeline selected during validation."""

    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Validation report not found: {RESULTS_PATH}")

    if not SELECTED_PIPELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Selected validation pipeline not found: {SELECTED_PIPELINE_PATH}"
        )

    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    if report.get("status") != "validation_only":
        raise ThresholdAuditError("Validation report status is not validation_only.")

    if report.get("test_split_evaluated") is not False:
        raise ThresholdAuditError("Validation report indicates test-set evaluation.")

    expected_hash = str(report.get("selected_pipeline_sha256", ""))
    actual_hash = calculate_file_hash(SELECTED_PIPELINE_PATH, "sha256")

    if actual_hash != expected_hash:
        raise ThresholdAuditError(
            "Selected validation pipeline SHA-256 mismatch. "
            f"Expected {expected_hash}, found {actual_hash}."
        )

    pipeline = joblib.load(SELECTED_PIPELINE_PATH)
    classes = tuple(int(value) for value in pipeline.classes_.tolist())

    if classes != (0, 1):
        raise ThresholdAuditError(f"Expected pipeline classes (0, 1), found {classes}.")

    return report, pipeline


def build_validation_predictions(
    data: pd.DataFrame,
    pipeline: object,
    feature_mode: FeatureMode,
) -> pd.DataFrame:
    """Returns one audited prediction row for each locked validation record."""

    validation = data.loc[data[SPLIT_COLUMN].eq(VALIDATION_SPLIT)].copy()

    if validation.empty:
        raise ThresholdAuditError("The locked validation split contains no records.")

    features = make_feature_series(validation, feature_mode)
    probabilities = np.asarray(pipeline.predict_proba(features), dtype=float)
    classes = [int(value) for value in pipeline.classes_.tolist()]

    if probabilities.shape != (len(validation), 2):
        raise ThresholdAuditError(
            f"Expected probability shape ({len(validation)}, 2), "
            f"found {probabilities.shape}."
        )

    class_indices = {label: index for index, label in enumerate(classes)}
    fake_probability = probabilities[:, class_indices[0]]
    real_probability = probabilities[:, class_indices[1]]
    predicted_label = np.where(real_probability >= fake_probability, 1, 0).astype(int)
    confidence = np.maximum(fake_probability, real_probability)
    actual_label = validation[LABEL_COLUMN].to_numpy(dtype=int)

    return pd.DataFrame(
        {
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "confidence": confidence,
            "correct": predicted_label == actual_label,
        }
    )


def audit_confidence_buckets(
    predictions: pd.DataFrame,
    edges: Iterable[float] = CONFIDENCE_EDGES,
) -> list[dict[str, object]]:
    """Measures observed correctness inside fixed predicted-confidence ranges."""

    edge_values = list(edges)
    rows: list[dict[str, object]] = []

    for lower, upper in zip(edge_values[:-1], edge_values[1:], strict=True):
        inclusive_upper = upper > 1.0
        mask = predictions["confidence"].ge(lower) & (
            predictions["confidence"].le(1.0)
            if inclusive_upper
            else predictions["confidence"].lt(upper)
        )
        group = predictions.loc[mask]
        count = int(len(group))
        display_upper = 1.0 if inclusive_upper else upper

        rows.append(
            {
                "lower_bound": round(lower, 2),
                "upper_bound": round(display_upper, 2),
                "range": f"{lower:.2f}-{display_upper:.2f}",
                "records": count,
                "share": _safe_ratio(count, len(predictions)),
                "mean_confidence": _safe_mean(group["confidence"]),
                "observed_accuracy": _safe_mean(group["correct"]),
                "calibration_gap": _safe_gap(group),
                "errors": int((~group["correct"]).sum()) if count else 0,
            }
        )

    return rows


def audit_threshold_policy(
    predictions: pd.DataFrame,
    thresholds: Iterable[float] = THRESHOLDS,
) -> list[dict[str, object]]:
    """Measures three-state abstention policies at each symmetric threshold."""

    rows: list[dict[str, object]] = []
    total = len(predictions)

    for threshold in thresholds:
        decisive = predictions["confidence"].ge(threshold)
        decided = predictions.loc[decisive]
        uncertain = total - len(decided)
        fake_decisions = decided.loc[decided["predicted_label"].eq(0)]
        real_decisions = decided.loc[decided["predicted_label"].eq(1)]

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "total_records": int(total),
                "decisive_records": int(len(decided)),
                "uncertain_records": int(uncertain),
                "coverage": _safe_ratio(len(decided), total),
                "uncertain_rate": _safe_ratio(uncertain, total),
                "selective_accuracy": _safe_mean(decided["correct"]),
                "decisive_errors": int((~decided["correct"]).sum()),
                "fake_decisions": _class_decision_summary(fake_decisions, 0),
                "real_decisions": _class_decision_summary(real_decisions, 1),
            }
        )

    return rows


def audit_calibration(predictions: pd.DataFrame) -> dict[str, object]:
    """Audits probability quality for class 1 and top-class confidence."""

    actual = predictions["actual_label"].to_numpy(dtype=int)
    real_probability = predictions["real_probability"].to_numpy(dtype=float)
    fraction_positive, mean_predicted = calibration_curve(
        actual,
        real_probability,
        n_bins=CALIBRATION_BINS,
        strategy="uniform",
    )

    bins = [
        {
            "mean_real_probability": round(float(predicted), 8),
            "observed_real_frequency": round(float(observed), 8),
            "absolute_gap": round(abs(float(predicted) - float(observed)), 8),
        }
        for predicted, observed in zip(
            mean_predicted,
            fraction_positive,
            strict=True,
        )
    ]

    return {
        "brier_score": round(
            float(brier_score_loss(actual, real_probability)),
            8,
        ),
        "log_loss": round(
            float(log_loss(actual, np.column_stack((1.0 - real_probability, real_probability)))),
            8,
        ),
        "expected_calibration_error_top_class": _top_class_ece(predictions),
        "bins_for_real_label_probability": bins,
    }


def build_baseline_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Records the unchanged default 0.50 classification behaviour."""

    actual = predictions["actual_label"]
    predicted = predictions["predicted_label"]
    macro = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    return {
        "records": int(len(predictions)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 8),
        "macro_precision": round(float(macro[0]), 8),
        "macro_recall": round(float(macro[1]), 8),
        "macro_f1": round(float(macro[2]), 8),
        "mean_confidence": round(float(predictions["confidence"].mean()), 8),
        "minimum_confidence": round(float(predictions["confidence"].min()), 8),
        "maximum_confidence": round(float(predictions["confidence"].max()), 8),
    }


def select_policy_examples(
    threshold_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Surfaces evidence at common candidate thresholds without selecting one."""

    candidates = {0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
    return {
        "selection_status": "not_selected",
        "note": (
            "These are validation-only measurements. "
            "No runtime threshold or display label has been changed."
        ),
        "candidate_rows": [
            row for row in threshold_rows if float(row["threshold"]) in candidates
        ],
    }


def run_threshold_audit() -> dict[str, object]:
    """Runs and persists the locked validation-only confidence audit."""

    experiment_report, pipeline = load_locked_validation_pipeline()
    data, preparation_metadata = load_experiment_data()
    feature_mode = experiment_report.get("selected_feature_mode")

    if feature_mode not in {"title", "text", "combined"}:
        raise ExperimentError(f"Unsupported selected feature mode: {feature_mode!r}")

    predictions = build_validation_predictions(data, pipeline, feature_mode)
    confidence_buckets = audit_confidence_buckets(predictions)
    threshold_policy = audit_threshold_policy(predictions)
    calibration = audit_calibration(predictions)

    report: dict[str, object] = {
        "status": "validation_threshold_audit_complete",
        "scope": "validation_only",
        "test_split_used": False,
        "model_changed": False,
        "runtime_threshold_changed": False,
        "selected_feature_mode": feature_mode,
        "selected_pipeline_sha256": experiment_report["selected_pipeline_sha256"],
        "baseline_default_decision": build_baseline_metrics(predictions),
        "confidence_buckets": confidence_buckets,
        "threshold_policy": threshold_policy,
        "calibration": calibration,
        "policy_review": select_policy_examples(threshold_policy),
        "dataset_preparation": preparation_metadata,
    }

    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_threshold_csv(threshold_policy)
    _write_flat_csv(CONFIDENCE_BUCKET_CSV_PATH, confidence_buckets)
    _write_flat_csv(
        CALIBRATION_CSV_PATH,
        calibration["bins_for_real_label_probability"],
    )
    return report


def _class_decision_summary(
    group: pd.DataFrame,
    predicted_label: int,
) -> dict[str, object]:
    count = len(group)
    correct = group["actual_label"].eq(predicted_label)
    return {
        "predicted_label": predicted_label,
        "records": int(count),
        "precision_within_decisions": _safe_mean(correct),
        "errors": int((~correct).sum()) if count else 0,
        "mean_confidence": _safe_mean(group["confidence"]),
    }


def _top_class_ece(predictions: pd.DataFrame) -> float | None:
    weighted_gap = 0.0
    total = len(predictions)

    if total == 0:
        return None

    for row in audit_confidence_buckets(predictions):
        if int(row["records"]) == 0:
            continue
        weighted_gap += float(row["share"]) * abs(float(row["calibration_gap"]))

    return round(weighted_gap, 8)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator / denominator), 8)


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.astype(float).mean()), 8)


def _safe_gap(group: pd.DataFrame) -> float | None:
    if group.empty:
        return None
    return round(
        float(group["correct"].astype(float).mean() - group["confidence"].mean()),
        8,
    )


def _write_flat_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_threshold_csv(rows: list[dict[str, object]]) -> None:
    flattened: list[dict[str, object]] = []

    for row in rows:
        fake = row["fake_decisions"]
        real = row["real_decisions"]
        flattened.append(
            {
                "threshold": row["threshold"],
                "total_records": row["total_records"],
                "decisive_records": row["decisive_records"],
                "uncertain_records": row["uncertain_records"],
                "coverage": row["coverage"],
                "uncertain_rate": row["uncertain_rate"],
                "selective_accuracy": row["selective_accuracy"],
                "decisive_errors": row["decisive_errors"],
                "fake_records": fake["records"],
                "fake_precision": fake["precision_within_decisions"],
                "fake_errors": fake["errors"],
                "real_records": real["records"],
                "real_precision": real["precision_within_decisions"],
                "real_errors": real["errors"],
            }
        )

    _write_flat_csv(THRESHOLD_CSV_PATH, flattened)


def main() -> None:
    print("Running validation-only confidence and threshold audit...")
    report = run_threshold_audit()
    print(json.dumps(report["baseline_default_decision"], indent=2))
    print("\nCandidate threshold policies:")

    for row in report["policy_review"]["candidate_rows"]:
        print(
            f"  threshold={row['threshold']:.2f} "
            f"coverage={row['coverage']:.4f} "
            f"uncertain={row['uncertain_rate']:.4f} "
            f"selective_accuracy={row['selective_accuracy']:.4f}"
        )

    print(f"\nAudit JSON: {AUDIT_JSON_PATH}")
    print(f"Threshold CSV: {THRESHOLD_CSV_PATH}")
    print(f"Confidence buckets CSV: {CONFIDENCE_BUCKET_CSV_PATH}")
    print(f"Calibration bins CSV: {CALIBRATION_CSV_PATH}")
    print("No model, runtime threshold, label, or held-out test result was changed.")


if __name__ == "__main__":
    main()
